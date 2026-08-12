//! Pure diffing helpers used to show a "golden diff" before a mutating write.
//! Kept dependency-free so `core-domain` stays clean of infra.

/// Render a compact line-diff between two file contents using a longest-common-
/// subsequence over lines. Added lines are prefixed `+`, removed lines `-`,
/// and a small window of unchanged context lines is shown on either side so the
/// reviewer (human user) sees exactly what would change.
///
/// Returns an empty string when there is no difference.
pub fn generate_diff(old: &str, new: &str) -> String {
    if old == new {
        return String::new();
    }
    let old_lines: Vec<&str> = old.split('\n').collect();
    let new_lines: Vec<&str> = new.split('\n').collect();

    // LCS DP table.
    let (n, m) = (old_lines.len(), new_lines.len());
    let mut table = vec![vec![0usize; m + 1]; n + 1];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            table[i][j] = if old_lines[i] == new_lines[j] {
                table[i + 1][j + 1] + 1
            } else {
                table[i + 1][j].max(table[i][j + 1])
            };
        }
    }

    // Walk the table to produce an edit script.
    let mut ops: Vec<(char, String)> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < n && j < m {
        if old_lines[i] == new_lines[j] {
            ops.push((' ', old_lines[i].to_string()));
            i += 1;
            j += 1;
        } else if table[i + 1][j] >= table[i][j + 1] {
            ops.push(('-', old_lines[i].to_string()));
            i += 1;
        } else {
            ops.push(('+', new_lines[j].to_string()));
            j += 1;
        }
    }
    while i < n {
        ops.push(('-', old_lines[i].to_string()));
        i += 1;
    }
    while j < m {
        ops.push(('+', new_lines[j].to_string()));
        j += 1;
    }

    render_with_context(&ops, 2)
}

/// Collapse an edit script into a readable preview, keeping `ctx` unchanged
/// lines of context around each change.
fn render_with_context(ops: &[(char, String)], ctx: usize) -> String {
    // First collapse runs of context lines into "unchanged" markers whose range
    // we can later expand.
    let changed: Vec<bool> = ops.iter().map(|(c, _)| *c != ' ').collect();
    let mut out = String::new();
    let len = ops.len();
    let mut k = 0usize;
    while k < len {
        if changed[k] {
            // Emit a change run (with up to `ctx` preceding context already
            // handled by the previous bucket).
            let start = k;
            while k < len && changed[k] {
                k += 1;
            }
            let end = k;
            // Context before the run.
            let pre_start = start.saturating_sub(ctx);
            let mut i = pre_start;
            while i < start {
                out.push(' ');
                out.push_str(&ops[i].1);
                out.push('\n');
                i += 1;
            }
            for op in &ops[start..end] {
                out.push(op.0);
                out.push_str(&op.1);
                out.push('\n');
            }
            // Context after the run.
            let post_end = (end + ctx).min(len);
            let mut i = end;
            while i < post_end {
                out.push(' ');
                out.push_str(&ops[i].1);
                out.push('\n');
                i += 1;
            }
        } else {
            k += 1;
        }
    }
    // Trim any leading/trailing blank lines for neatness.
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_content_yields_empty_diff() {
        assert_eq!(generate_diff("a\nb\nc", "a\nb\nc"), "");
    }

    #[test]
    fn insertion_is_marked_with_plus() {
        let d = generate_diff("a\nc", "a\nb\nc");
        assert!(d.contains("+b"), "diff should show added line: {d}");
    }

    #[test]
    fn deletion_is_marked_with_minus() {
        let d = generate_diff("a\nb\nc", "a\nc");
        assert!(d.contains("-b"), "diff should show removed line: {d}");
    }

    #[test]
    fn change_shows_both_sides() {
        let d = generate_diff("old\nvalue", "new\nvalue");
        assert!(d.contains("-old"));
        assert!(d.contains("+new"));
    }
}
