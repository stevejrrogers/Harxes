//! The agent's plan/todo list. Kept in core-domain because plan progress is
//! part of the domain model. Invariant: at most one item is `InProgress` at a
//! time — enforce it with [`normalize_todos`] whenever a whole list is written.

use serde::{Deserialize, Serialize};

/// Lifecycle state of one plan item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl TodoStatus {
    pub fn parse(raw: &str) -> Option<TodoStatus> {
        match raw.trim().to_lowercase().as_str() {
            "pending" => Some(TodoStatus::Pending),
            "in_progress" | "in-progress" | "inprogress" => Some(TodoStatus::InProgress),
            "completed" | "done" => Some(TodoStatus::Completed),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            TodoStatus::Pending => "pending",
            TodoStatus::InProgress => "in_progress",
            TodoStatus::Completed => "completed",
        }
    }
}

/// One plan item. `active_form` is the present-continuous phrasing shown while
/// the item is in progress (e.g. label "Run tests" → active form "Running
/// tests"); it falls back to the label when empty.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TodoItem {
    pub label: String,
    #[serde(default)]
    pub active_form: String,
    pub status: TodoStatus,
}

impl TodoItem {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            active_form: String::new(),
            status: TodoStatus::Pending,
        }
    }

    pub fn done(&self) -> bool {
        self.status == TodoStatus::Completed
    }

    /// The phrasing to show while this item is being worked on.
    pub fn active_label(&self) -> &str {
        if self.active_form.trim().is_empty() {
            &self.label
        } else {
            &self.active_form
        }
    }

    /// Render this item as a checklist line, e.g. `- [x] Finish auth`.
    /// In-progress items are marked `[~]`.
    pub fn render(&self) -> String {
        let mark = match self.status {
            TodoStatus::Pending => " ",
            TodoStatus::InProgress => "~",
            TodoStatus::Completed => "x",
        };
        format!("- [{}] {}", mark, self.label)
    }
}

/// Render a list of items as a markdown checklist block.
pub fn render_todos(items: &[TodoItem]) -> String {
    if items.is_empty() {
        return String::new();
    }
    items.iter().map(TodoItem::render).collect::<Vec<_>>().join("\n")
}

/// Enforce the single-in-progress invariant: every `InProgress` item after the
/// first is demoted back to `Pending`.
pub fn normalize_todos(items: &mut [TodoItem]) {
    let mut seen_active = false;
    for item in items.iter_mut() {
        if item.status == TodoStatus::InProgress {
            if seen_active {
                item.status = TodoStatus::Pending;
            }
            seen_active = true;
        }
    }
}

/// The operations the agent can perform on the plan via the `Todo` tool.
/// `Write` replaces the whole list atomically (no index bookkeeping for the
/// model); `List` reads it back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TodoAction {
    List,
    Write,
}

impl TodoAction {
    pub fn parse(raw: &str) -> Option<TodoAction> {
        match raw.trim().to_lowercase().as_str() {
            "list" => Some(TodoAction::List),
            "write" => Some(TodoAction::Write),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            TodoAction::List => "list",
            TodoAction::Write => "write",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_marks_each_status() {
        let items = vec![
            TodoItem::new("first"),
            TodoItem { label: "second".into(), active_form: String::new(), status: TodoStatus::InProgress },
            TodoItem { label: "third".into(), active_form: String::new(), status: TodoStatus::Completed },
        ];
        let out = render_todos(&items);
        assert!(out.contains("- [ ] first"));
        assert!(out.contains("- [~] second"));
        assert!(out.contains("- [x] third"));
    }

    #[test]
    fn normalize_keeps_only_first_in_progress() {
        let mut items = vec![
            TodoItem { label: "a".into(), active_form: String::new(), status: TodoStatus::InProgress },
            TodoItem { label: "b".into(), active_form: String::new(), status: TodoStatus::InProgress },
        ];
        normalize_todos(&mut items);
        assert_eq!(items[0].status, TodoStatus::InProgress);
        assert_eq!(items[1].status, TodoStatus::Pending);
    }

    #[test]
    fn active_label_falls_back_to_label() {
        let mut i = TodoItem::new("Run tests");
        assert_eq!(i.active_label(), "Run tests");
        i.active_form = "Running tests".into();
        assert_eq!(i.active_label(), "Running tests");
    }

    #[test]
    fn action_and_status_parse() {
        assert_eq!(TodoAction::parse("WRITE"), Some(TodoAction::Write));
        assert_eq!(TodoAction::parse("list"), Some(TodoAction::List));
        assert_eq!(TodoAction::parse("bogus"), None);
        assert_eq!(TodoStatus::parse("in_progress"), Some(TodoStatus::InProgress));
        assert_eq!(TodoStatus::parse("done"), Some(TodoStatus::Completed));
    }
}
