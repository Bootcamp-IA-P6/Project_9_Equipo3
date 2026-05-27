-- =====================================================================
-- SignalMod — Predictions table setup
-- Run this in Supabase SQL Editor:
--   Dashboard → SQL Editor → New query → paste this → Run
-- =====================================================================

-- 1. Table
create table if not exists public.predictions (
    id          bigserial primary key,
    created_at  timestamptz not null default now(),
    text        text not null,
    video_id    text,
    video_url   text,
    probability double precision,
    is_toxic    boolean,
    labels      text[] default '{}',
    model_used  text,
    threshold   double precision,
    latency_ms  double precision,
    source      text,         -- "api_direct" | "video_fetch" | "user_comment"
    author      text
);

-- Migration: add author column on existing installs
alter table public.predictions add column if not exists author text;

-- 2. Indexes for the queries the API will run
create index if not exists predictions_created_at_idx
    on public.predictions (created_at desc);

create index if not exists predictions_video_id_idx
    on public.predictions (video_id);

-- 3. Row Level Security: allow anonymous insert + select
--    (we are using the publishable key from the frontend / backend with no auth)
alter table public.predictions enable row level security;

drop policy if exists "anon_insert" on public.predictions;
create policy "anon_insert"
    on public.predictions
    for insert
    to anon
    with check (true);

drop policy if exists "anon_select" on public.predictions;
create policy "anon_select"
    on public.predictions
    for select
    to anon
    using (true);

-- 4. Sanity check (run separately if you want)
-- select count(*) from public.predictions;
