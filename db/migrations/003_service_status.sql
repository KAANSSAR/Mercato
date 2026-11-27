-- Single-row table the site reads to decide whether live features are shown.

create table if not exists service_status (
    id               int primary key default 1 check (id = 1),
    plan             text not null,
    live_enabled     boolean not null,
    seasons          int[] not null,         -- seasons the API plan can reach
    reason           text,                   -- shown on the site when live is off
    requests_today   int,
    request_limit    int,
    checked_at       timestamptz not null default now(),
    changed_at       timestamptz not null default now(),  -- last time live_enabled flipped
    last_live_update timestamptz             -- last successful post-match pull
);
