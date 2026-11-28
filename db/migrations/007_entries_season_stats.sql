-- Who played in which competition each season. Promotion, relegation and European
-- qualification all fall out of this, instead of a fixed "current" flag.
create table if not exists competition_entries (
    competition_id int not null references competitions(id),
    season_id      int not null references seasons(id),
    club_id        int not null references clubs(id),
    primary key (competition_id, season_id, club_id)
);
create index if not exists entries_club on competition_entries (club_id);

-- API-Football season totals per player and team. Also the training source for seasons
-- not backfilled match by match.
create table if not exists api_player_season (
    api_player_id  int  not null,
    competition_id int  not null references competitions(id),
    season_id      int  not null references seasons(id),
    team_api_id    int  not null,
    mercato_id     int  references players(mercato_id),
    minutes        int,
    appearances    int,
    position       text,
    rating         real,
    stats          jsonb not null,
    primary key (api_player_id, competition_id, season_id, team_api_id)
);
create index if not exists aps_player on api_player_season (mercato_id);
