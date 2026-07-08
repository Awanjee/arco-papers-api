-- 003_enable_rls_all_tables.sql
--
-- Closes the public RLS exposure flagged by Supabase's security advisor on
-- 2026-07-07 (all 13 public tables had RLS disabled, meaning anyone with the
-- project URL could read/write via PostgREST using the anon key).
--
-- This migration enables RLS with NO policies on every public table. That is
-- intentional and correct here, not a placeholder to fill in later:
--   - The only consumer of this database is istatis-papers-api, running
--     server-side on Render.
--   - As of this same change, database.py was switched from the anon key
--     (SUPABASE_KEY) to the service_role key (SUPABASE_SERVICE_KEY).
--   - The service_role key bypasses RLS entirely, so the API keeps working
--     exactly as before.
--   - With RLS on and zero policies, the anon and authenticated roles get a
--     hard deny on every table. No Flutter client or browser can hit these
--     tables directly through Supabase, even with the anon key.
--
-- Do NOT re-introduce a client-side Supabase key against these tables
-- without writing real RLS policies (tenant_id scoping, ownership checks,
-- etc) first. Safe to re-run: ENABLE ROW LEVEL SECURITY is idempotent.

ALTER TABLE public.tenants ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.products ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pricing_tiers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.quotes ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.order_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.parties ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.party_aliases ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.document_extractions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.transaction_line_items ENABLE ROW LEVEL SECURITY;
