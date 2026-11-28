-- Valuations outside the model's data: shown with their range and a label, kept out of
-- the under/overvalued rankings.
alter table valuations add column if not exists capped boolean not null default false;
alter table valuations add column if not exists low_confidence boolean not null default false;
