-- Create the dashboard_logs database and demo_user role used by
-- CaberUserConsole.
-- Runs once on first postgres container init via /docker-entrypoint-initdb.d/.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'demo_user') THEN
        CREATE ROLE demo_user WITH LOGIN PASSWORD 'demo_password';
    END IF;
END
$$;

CREATE DATABASE dashboard_logs OWNER demo_user;
