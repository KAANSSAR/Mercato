-- Current-squad flags, ClubElo history and Understat season stats.

alter table clubs   add column if not exists current_top5 boolean not null default false;
alter table players add column if not exists current_top5 boolean not null default false;

-- ClubElo publishes ratings as date ranges. Mapped to clubs via clubs.clubelo_name.
create table if not exists elo_history (
    clubelo_name text not null,
    country      text,
    level        int,
    elo          real not null,
    valid_from   date not null,
    valid_to     date not null,
    primary key (clubelo_name, valid_from)
);
create index if not exists elo_history_lookup on elo_history (clubelo_name, valid_from, valid_to);

-- Understat player-season stats (top 5 leagues, 2014/15 on). Mapped via id_map.understat_id.
create table if not exists understat_player_season (
    understat_player_id int  not null,
    season              int  not null,  -- start year
    league              text not null,  -- GB1, ES1, IT1, L1, FR1
    team                text not null,
    player              text not null,
    position            text,
    matches             int,
    minutes             int,
    goals               int,
    np_goals            int,
    xg                  real,
    np_xg               real,
    assists             int,
    xa                  real,
    shots               int,
    key_passes          int,
    yellow_cards        int,
    red_cards           int,
    xg_chain            real,
    xg_buildup          real,
    primary key (understat_player_id, season, league, team)
);
