# web

Next.js frontend, scaffolded in the week of 21 Dec. Deployed to Vercel at mercato.kanishka-sarkar.com.

## Live status contract

The backend keeps one row in `service_status`. Every page reads it (cached with the page) and adapts. Nothing on the site should ever error because live data is missing.

| Column | Use |
|---|---|
| `live_enabled` | Master switch for live features |
| `reason` | Banner text when live is off |
| `last_live_update` | "Updated 2h ago" and "Stats as of 22 Nov" |
| `seasons` | Which seasons have full stats, for season pickers |
| `plan` | Not shown publicly. Free plan still runs live updates |

### When `live_enabled` is true
Everything shows as normal. Show "Updated Xh ago" from `last_live_update`. If that's more than 48 hours old on a matchday, show a soft "Updates delayed" note.

### When `live_enabled` is false
- **Banner** at the top of every page: `reason`, plus "Stats as of <last_live_update date>". Neutral tone, not an error style.
- **Hide:** "updated after every game" badges, latest results strip, live form strip, "this week's matches", standings "as of today" wording.
- **Keep, from last good data:** player and club profiles, valuations and ranges, SHAP drivers, the under/overvalued ranking, compare, twin finder, historical match logs.
- **Relabel:** standings and form show "as of <date>" instead of "current".
- Season pickers only offer seasons in `seasons` plus any already stored.

### Rules
- Gate live components with one shared hook or component, e.g. `<LiveOnly>`, never ad hoc checks.
- If `service_status` can't be read, treat it as live off. Fail closed, never show a broken widget.
