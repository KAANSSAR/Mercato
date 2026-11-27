-- Run after schema.sql. Makes loaders idempotent and adds Transfermarkt competition codes.

alter table competitions add column if not exists tm_code text unique;  -- GB1, ES1, CL ...
alter table competitions alter column api_football_id drop not null;

alter table transfers add column if not exists source_key text unique;   -- e.g. tm:<player>:<date>:<from>:<to>

alter table clubs add column if not exists tm_competition text;          -- domestic league code

alter table players add column if not exists sub_position text;          -- raw Transfermarkt label
alter table players add column if not exists foot text;
alter table players add column if not exists height_cm int;
