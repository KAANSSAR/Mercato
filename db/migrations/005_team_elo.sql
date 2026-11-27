-- Mercato's own Elo replaces ClubElo (public feed closed late 2026).

drop table if exists elo_history;
drop table if exists elo_ratings;

-- One row per club per game. Keyed on Transfermarkt ids, so every opponent is covered,
-- including foreign clubs that aren't in the clubs table.
create table if not exists team_elo (
    game_id             bigint not null,   -- Transfermarkt game id
    tm_club_id          int    not null,
    opponent_tm_club_id int    not null,
    date                date   not null,
    competition_id      text   not null,
    is_home             boolean not null,
    elo_pre             real   not null,
    opp_elo_pre         real   not null,
    elo_post            real   not null,
    expected            real   not null,
    result              real   not null,   -- 1 win, 0.5 draw, 0 loss
    primary key (game_id, tm_club_id)
);
create index if not exists team_elo_club_date on team_elo (tm_club_id, date);

-- Latest rating per club, for club pages and the matchday weights
create or replace view club_elo_latest as
select distinct on (tm_club_id) tm_club_id, date, elo_post as elo
from team_elo
order by tm_club_id, date desc, game_id desc;
