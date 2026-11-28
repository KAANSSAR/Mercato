-- Images and flags for the site. photo_url and logo_url point at our own copies in
-- Supabase Storage (bucket "media"), or at the source when storage isn't set up.
-- flag_code follows flag-icons (es, gb-eng, ...), which the site bundles.
alter table players      add column if not exists photo_url text;
alter table clubs        add column if not exists logo_url text;
alter table competitions add column if not exists logo_url text;
alter table nations      add column if not exists flag_code text;
alter table clubs        add column if not exists colours_from text;  -- crest, manual
alter table nations      add column if not exists colours_from text;  -- flag, manual
