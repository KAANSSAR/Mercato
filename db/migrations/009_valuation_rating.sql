-- Model v2: the performance rating (0 to 100 within position) and its components,
-- stored next to each valuation so player pages can show why.
alter table valuations add column if not exists rating real;
alter table valuations add column if not exists components jsonb;
