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

    #[tokio::test]
    async fn rejects_non_http_schemes() {
        let w = ReqwestWeb::default();
        assert!(w.fetch("file:///etc/passwd").await.is_err());
        assert!(w.fetch("ftp://x").await.is_err());
    }
}
