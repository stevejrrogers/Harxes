//! On-demand skills: packaged instruction sets the agent loads only when a
//! task needs them. A skill is a directory `<root>/skills/<name>/SKILL.md`
//! with YAML-ish frontmatter (`name`, `description`) and a markdown body.
//!
//! Skills are discovered under both `.harxes/skills/` and `.claude/skills/`
//! (the latter for cross-tool compatibility with Claude Code). Only the
//! one-line descriptions are injected into the system prompt; the full body is
//! fetched by the `Skill` tool when the model decides a skill applies.

use std::path::{Path, PathBuf};

/// Metadata for one discovered skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    /// Path to the SKILL.md file.
    pub path: PathBuf,
}

/// The directories skills live under, relative to the project root, in
/// precedence order (project-local first).
fn skill_roots(project_root: &Path) -> Vec<PathBuf> {
    vec![
        project_root.join(".harxes").join("skills"),
        project_root.join(".claude").join("skills"),
    ]
}

/// Split SKILL.md content into (name, description, body). Frontmatter is an
/// optional leading `---` fenced block of `key: value` lines; `name`/
/// `description` are read from it, and the body is everything after.
pub fn parse_skill(content: &str, fallback_name: &str) -> (String, String, String) {
    let mut name = fallback_name.to_string();
    let mut description = String::new();
    let body;
    let trimmed = content.trim_start();
    if let Some(rest) = trimmed.strip_prefix("---") {
        // Frontmatter block ends at the next line that is exactly `---`.
        if let Some(end) = rest.find("\n---") {
            let front = &rest[..end];
            for line in front.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let (k, v) = (k.trim().to_lowercase(), v.trim().trim_matches(['"', '\'']));
                    match k.as_str() {
                        "name" if !v.is_empty() => name = v.to_string(),
                        "description" if !v.is_empty() => description = v.to_string(),
                        _ => {}
                    }
                }
            }
            // Body starts after the closing `---` line.
            let after = &rest[end + 4..];
            body = after.trim_start_matches(['\r', '\n']).to_string();
        } else {
            body = content.to_string();
        }
    } else {
        body = content.to_string();
    }
    (name, description, body)
}

/// Discover all skills under the project root, de-duplicated by name (a
/// `.harxes` skill shadows a `.claude` one of the same name).
pub fn discover(project_root: &Path) -> Vec<SkillMeta> {
    let mut out: Vec<SkillMeta> = Vec::new();
    for root in skill_roots(project_root) {
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        for e in entries.flatten() {
            if !e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let dir_name = e.file_name().to_string_lossy().into_owned();
            let path = e.path().join("SKILL.md");
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            let (name, description, _body) = parse_skill(&content, &dir_name);
            if out.iter().any(|s| s.name == name) {
                continue; // earlier root wins
            }
            out.push(SkillMeta {
                name,
                description,
                path,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Load the full body of a skill by name (frontmatter stripped), searching the
/// same roots. Returns `None` if no skill by that name exists.
pub fn load_body(project_root: &Path, name: &str) -> Option<String> {
    let want = name.trim();
    for root in skill_roots(project_root) {
        // Prefer an exact directory match, then any skill whose frontmatter
        // name matches.
        let direct = root.join(want).join("SKILL.md");
        if let Ok(content) = std::fs::read_to_string(&direct) {
            return Some(parse_skill(&content, want).2);
        }
    }
    // Fall back to a frontmatter-name match across all discovered skills.
    for s in discover(project_root) {
        if s.name == want {
            if let Ok(content) = std::fs::read_to_string(&s.path) {
                return Some(parse_skill(&content, &s.name).2);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_and_body() {
        let md = "---\nname: deploy-web\ndescription: How to ship the web app\n---\n\nStep 1. do X\nStep 2. do Y\n";
        let (name, desc, body) = parse_skill(md, "fallback");
        assert_eq!(name, "deploy-web");
        assert_eq!(desc, "How to ship the web app");
        assert!(body.starts_with("Step 1"));
        assert!(!body.contains("description:"));
    }

    #[test]
    fn no_frontmatter_keeps_body_and_fallback_name() {
        let (name, desc, body) = parse_skill("just instructions", "my-skill");
        assert_eq!(name, "my-skill");
        assert_eq!(desc, "");
        assert_eq!(body, "just instructions");
    }

    #[test]
    fn discover_and_load_roundtrip() {
        let root = std::env::temp_dir().join(format!("hx-skills-{}", std::process::id()));
        let dir = root.join(".harxes").join("skills").join("review");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("SKILL.md"),
            "---\nname: review\ndescription: code review checklist\n---\nCheck tests.\n",
        )
        .unwrap();
        let found = discover(&root);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "review");
        assert_eq!(found[0].description, "code review checklist");
        let body = load_body(&root, "review").unwrap();
        assert!(body.contains("Check tests."));
        assert!(load_body(&root, "nope").is_none());
    }
}
