//! Driven-port adapter (hexagonal infrastructure ring): fetches web pages for
//! the agent's `Fetch` tool. HTML is reduced to readable text; responses are
//! size-capped so one page cannot flood the model's context.

use async_trait::async_trait;
use harxes_core_domain::ports::WebPort;

const MAX_BODY_BYTES: usize = 2_000_000;
const MAX_TEXT_CHARS: usize = 20_000;
const TIMEOUT_SECS: u64 = 30;

/// reqwest-backed [`WebPort`] implementation.
pub struct ReqwestWeb {
    http: reqwest::Client,
}

impl Default for ReqwestWeb {
    fn default() -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(TIMEOUT_SECS))
                .user_agent(concat!("harxes/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("reqwest client"),
        }
    }
}

#[async_trait]
impl WebPort for ReqwestWeb {
    async fn fetch(&self, url: &str) -> Result<String, String> {
        let url = url.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(format!("only http(s) URLs are supported, got '{url}'"));
        }
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();
        if !status.is_success() {
            return Err(format!("HTTP {status} for {url}"));
        }
        let body = resp
            .bytes()
            .await
            .map_err(|e| format!("read failed: {e}"))?;
        let body = &body[..body.len().min(MAX_BODY_BYTES)];
        if !(content_type.contains("text")
            || content_type.contains("json")
            || content_type.contains("xml")
            || content_type.is_empty())
        {
            return Ok(format!(
                "[binary content: {} bytes of {content_type} — not shown]",
                body.len()
            ));
        }
        let raw = String::from_utf8_lossy(body);
        let text = if content_type.contains("html") || looks_like_html(&raw) {
            html_to_text(&raw)
        } else {
            raw.to_string()
        };
        let capped: String = text.chars().take(MAX_TEXT_CHARS).collect();
        if capped.len() < text.len() {
            Ok(format!("{capped}\n[content truncated at {MAX_TEXT_CHARS} chars]"))
        } else {
            Ok(capped)
        }
    }

    /// Key-free web search over DuckDuckGo's lite HTML endpoint.
    async fn search(&self, query: &str) -> Result<String, String> {
        let q = query.trim();
        if q.is_empty() {
            return Err("empty search query".to_string());
        }
        let url = format!("https://lite.duckduckgo.com/lite/?q={}", percent_encode(q));
        let html = self.fetch_raw(&url).await?;
        let results = parse_ddg_lite(&html, 8);
        if results.is_empty() {
            return Ok(format!("no results found for '{q}'"));
        }
        let mut out = String::new();
        for (i, (title, href, snippet)) in results.iter().enumerate() {
            out.push_str(&format!("{}. {title}\n   {href}\n", i + 1));
            if !snippet.trim().is_empty() {
                out.push_str(&format!("   {}\n", snippet.trim()));
            }
        }
        Ok(out.trim_end().to_string())
    }
}

impl ReqwestWeb {
    async fn fetch_raw(&self, url: &str) -> Result<String, String> {
        let resp = self
            .http
            .get(url)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {} for {url}", resp.status()));
        }
        let body = resp.bytes().await.map_err(|e| format!("read failed: {e}"))?;
        Ok(String::from_utf8_lossy(&body[..body.len().min(MAX_BODY_BYTES)]).to_string())
    }
}

fn percent_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Parse DuckDuckGo lite results: anchors with `rel="nofollow"` are result
/// links; the following `result-snippet` cell (when present) is the summary.
fn parse_ddg_lite(html: &str, max: usize) -> Vec<(String, String, String)> {
    let mut results = Vec::new();
    let mut rest = html;
    while results.len() < max {
        let Some(a) = rest.find("rel=\"nofollow\" href=\"") else { break };
        let after = &rest[a + "rel=\"nofollow\" href=\"".len()..];
        let Some(hend) = after.find('"') else { break };
        let href = &after[..hend];
        let after_tag = &after[hend..];
        let title = after_tag
            .find('>')
            .and_then(|o| {
                let t = &after_tag[o + 1..];
                t.find("</a>").map(|e| html_to_text(&t[..e]))
            })
            .unwrap_or_default();
        let tail = &after_tag[after_tag.find("</a>").map(|e| e + 4).unwrap_or(0)..];
        let snippet = tail
            .find("result-snippet")
            .and_then(|s| {
                let t = &tail[s..];
                let start = t.find('>')?;
                let end = t.find("</td>")?;
                Some(html_to_text(&t[start + 1..end]))
            })
            .unwrap_or_default();
        // Skip ad/redirect-less junk links.
        if href.starts_with("http") && !title.trim().is_empty() {
            results.push((title, href.to_string(), snippet));
        }
        rest = tail;
    }
    results
}

fn looks_like_html(s: &str) -> bool {
    let head = s.trim_start().get(..256.min(s.trim_start().len())).unwrap_or("");
    let h = head.to_lowercase();
    h.starts_with("<!doctype html") || h.starts_with("<html")
}

/// Reduce HTML to readable text: drop script/style blocks, strip tags, decode
/// common entities, collapse whitespace runs.
fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 4);
    let mut chars = html.char_indices().peekable();
    let lower = html.to_lowercase();
    let mut i = 0usize;
    let bytes = html.as_bytes();
    let _ = &mut chars;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip script/style wholesale.
            let mut skipped = false;
            for skip in ["script", "style"] {
                let open = format!("<{skip}");
                let close = format!("</{skip}>");
                if lower[i..].starts_with(&open) {
                    match lower[i..].find(&close) {
                        Some(end) => i += end + close.len(),
                        None => i = bytes.len(),
                    }
                    skipped = true;
                    break;
                }
            }
            if skipped {
                continue;
            }
            // Block-level tags become newlines so structure survives.
            for block in ["</p>", "<br", "</div>", "</h", "</li>", "</tr>", "</section>"] {
                if lower[i..].starts_with(block) {
                    out.push('\n');
                    break;
                }
            }
            match html[i..].find('>') {
                Some(end) => i += end + 1,
                None => break,
            }
        } else {
            let ch_end = html[i..]
                .char_indices()
                .nth(1)
                .map(|(o, _)| i + o)
                .unwrap_or(bytes.len());
            out.push_str(&html[i..ch_end]);
            i = ch_end;
        }
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'");
    // Collapse whitespace: max one blank line, trim line ends.
    let mut lines: Vec<String> = Vec::new();
    for line in decoded.lines() {
        let t = line.trim().to_string();
        if t.is_empty() {
            if lines.last().map(|l: &String| l.is_empty()).unwrap_or(true) {
                continue;
            }
            lines.push(String::new());
        } else {
            lines.push(t);
        }
    }
    lines.join("\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_reduces_to_text() {
        let html = r#"<!doctype html><html><head><title>T</title>
            <style>body{color:red}</style><script>alert(1)</script></head>
            <body><h1>Hello &amp; welcome</h1><p>First para</p>
            <p>Second&nbsp;para</p><div>tail</div></body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Hello & welcome"), "{text}");
        assert!(text.contains("First para"));
        assert!(text.contains("Second para"));
        assert!(!text.contains("alert(1)"), "script leaked: {text}");
        assert!(!text.contains("color:red"), "style leaked: {text}");
    }

    #[test]
    fn ddg_lite_results_parse() {
        let html = r#"<table>
          <tr><td>1.</td><td><a rel="nofollow" href="https://www.rust-lang.org/" class='result-link'>Rust Programming Language</a></td></tr>
          <tr><td></td><td class='result-snippet'>A language empowering everyone to build reliable software.</td></tr>
          <tr><td>2.</td><td><a rel="nofollow" href="https://doc.rust-lang.org/book/">The Rust Book</a></td></tr>
        </table>"#;
        let r = parse_ddg_lite(html, 8);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].0, "Rust Programming Language");
        assert_eq!(r[0].1, "https://www.rust-lang.org/");
        assert!(r[0].2.contains("empowering everyone"));
        assert_eq!(r[1].1, "https://doc.rust-lang.org/book/");
    }

    #[test]
    fn percent_encoding() {
        assert_eq!(percent_encode("rust async book"), "rust+async+book");
        assert_eq!(percent_encode("a&b=c"), "a%26b%3Dc");
    }

    #[tokio::test]
    async fn rejects_non_http_schemes() {
        let w = ReqwestWeb::default();
        assert!(w.fetch("file:///etc/passwd").await.is_err());
        assert!(w.fetch("ftp://x").await.is_err());
    }
}
