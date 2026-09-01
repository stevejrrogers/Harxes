//! Driven port (hexagonal): fetching web content so the agent can read
//! documentation and pages. Adapters handle HTTP, size caps and HTML→text.

use async_trait::async_trait;

/// A fetcher of web content, returning readable text.
#[async_trait]
pub trait WebPort: Send + Sync {
    /// Fetch `url` (http/https GET) and return readable text — HTML reduced
    /// to text, capped to a sane size. Errors come back as readable strings.
    async fn fetch(&self, url: &str) -> Result<String, String>;

    /// Search the web for `query`, returning a readable result list
    /// ("title — url" lines with snippets). Optional capability.
    async fn search(&self, _query: &str) -> Result<String, String> {
        Err("web search is not supported by this adapter".to_string())
    }
}
