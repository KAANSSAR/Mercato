-- Mercato schema (Postgres / Supabase)

-- Core
create table competitions (
    id              serial primary key,
    api_football_id int unique not null,
    name            text not null,
    kind            text not null check (kind in ('league', 'cup', 'international'))
);

create table seasons (
    id         serial primary key,
    year       int unique not null  -- season start year, 2025 = 2025/26
);

create table nations (
    id              serial primary key,
    name            text unique not null,
    colour_primary  text,  -- hex, e.g. #a50044
    colour_secondary text
);

create table clubs (
    id              serial primary key,
    api_football_id int unique,
    tm_club_id      int unique,
    clubelo_name    text unique,
    name            text not null,
    colour_primary  text,
    colour_secondary text
);

create table players (
    mercato_id     serial primary key,
    name           text not null,
    birth_date     date,
    nation_id      int references nations(id),
    club_id        int references clubs(id),
    position_group text check (position_group in ('GK','CB','FB','DM','CM','AM','W','ST')),
    contract_until date
);

-- One row per player, linking every source's ID
create table id_map (
    mercato_id      int primary key references players(mercato_id),
    api_football_id int unique,
    understat_id    int unique,
    tm_player_id    int unique,
    match_method    text,   -- exact, fuzzy, manual
    match_score     real
);

-- Events
create table matches (
    id              serial primary key,
    api_football_id int unique not null,
    competition_id  int not null references competitions(id),
    season_id       int not null references seasons(id),
    kickoff         timestamptz not null,
    home_club_id    int references clubs(id),
    away_club_id    int references clubs(id),
    home_goals      int,
    away_goals      int,
    status          text not null,  -- NS, FT, AET, PEN, ...
    stats_pulled_at timestamptz     -- null until stats ingested; idempotency key
);
create index on matches (status, stats_pulled_at);

create table player_match_stats (
    match_id      int references matches(id),
    mercato_id    int references players(mercato_id),
    club_id       int references clubs(id),
    minutes       int not null,
    stats         jsonb not null,  -- raw per-match stats from API-Football
    xg            real,            -- Understat, filled nightly
    xa            real,
    primary key (match_id, mercato_id)
);

create table team_match_stats (
    match_id   int references matches(id),
    club_id    int references clubs(id),
    possession real,
    stats      jsonb not null,
    primary key (match_id, club_id)
);

create table standings (
    competition_id int references competitions(id),
    season_id      int references seasons(id),
    club_id        int references clubs(id),
    as_of          date not null,
    position       int,
    points         int,
    played         int,
    goal_diff      int,
    primary key (competition_id, season_id, club_id, as_of)
);

create table elo_ratings (
    club_id int references clubs(id),
    as_of   date not null,
    elo     real not null,
    primary key (club_id, as_of)
);

-- Market
create table transfers (
    id            serial primary key,
    mercato_id    int references players(mercato_id),
    transfer_date date not null,
    from_club_id  int references clubs(id),
    to_club_id    int references clubs(id),
    fee_eur       numeric,  -- null for free / undisclosed
    source        text not null
);

create table tm_values (
    mercato_id int references players(mercato_id),
    as_of      date not null,
    value_eur  numeric not null,
    source     text not null,  -- dataset or on_demand
    primary key (mercato_id, as_of, source)
);

create table model_runs (
    id         serial primary key,
    run_at     timestamptz not null default now(),
    version    text not null,
    test_mdape real  -- median absolute % error on the test season
);

create table valuations (
    model_run_id int references model_runs(id),
    mercato_id   int references players(mercato_id),
    p10_eur      numeric not null,
    p50_eur      numeric not null,
    p90_eur      numeric not null,
    shap         jsonb not null,  -- {"age": 12000000, "xa_p90": 9000000, ...}
    primary key (model_run_id, mercato_id)
);

-- Ops
create table job_runs (
    id           serial primary key,
    job          text not null,
    started_at   timestamptz not null default now(),
    finished_at  timestamptz,
    requests     int default 0,
    rows_written int default 0,
    ok           boolean,
    error        text
);
