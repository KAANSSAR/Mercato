-- FIFA men's world ranking, one row per national team per official release (scraped from
-- the copy Global Football Rankings republishes). For weighting international games by
-- the opponent's strength.
create table if not exists fifa_ranking (
    release        date not null,
    team           text not null,
    rank           int not null,
    points         real not null,
    confederation  text,
    primary key (release, team)
);

create or replace view fifa_ranking_latest as
select * from fifa_ranking where release = (select max(release) from fifa_ranking);
