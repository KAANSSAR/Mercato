-- Global Football Rankings (globalfootballrankings.com): weekly snapshots of their league
-- and club ratings (0 to 100, 50 = world average), matched to our ids where possible.
create table if not exists gfr_league_ratings (
    snapshot          date not null,
    country           text not null,
    slug              text not null,
    name              text not null,
    rank              int,
    rating            real not null,
    tm_competition_id text,
    primary key (snapshot, country, slug)
);

create table if not exists gfr_club_ratings (
    snapshot    date not null,
    country     text not null,
    slug        text not null,
    name        text not null,
    rank        int,
    rating      real not null,
    tm_club_id  int,
    primary key (snapshot, country, slug)
);
create index if not exists gfr_club_ratings_tm on gfr_club_ratings (tm_club_id, snapshot);

-- latest snapshot, for the site and the model
create or replace view gfr_club_latest as
select * from gfr_club_ratings where snapshot = (select max(snapshot) from gfr_club_ratings);
create or replace view gfr_league_latest as
select * from gfr_league_ratings where snapshot = (select max(snapshot) from gfr_league_ratings);
