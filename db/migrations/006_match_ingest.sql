-- Fields used by match ingestion and in-house standings.

alter table matches add column if not exists round text;

alter table player_match_stats add column if not exists api_player_id int;
alter table player_match_stats add column if not exists position text;     -- G, D, M, F
alter table player_match_stats add column if not exists rating real;       -- API-Football match rating
alter table player_match_stats add column if not exists is_sub boolean;

alter table team_match_stats add column if not exists xg real;             -- API-Football team xG

alter table standings add column if not exists won int;
alter table standings add column if not exists drawn int;
alter table standings add column if not exists lost int;
alter table standings add column if not exists goals_for int;
alter table standings add column if not exists goals_against int;

create index if not exists matches_comp_season on matches (competition_id, season_id);
create index if not exists pms_player on player_match_stats (mercato_id);
