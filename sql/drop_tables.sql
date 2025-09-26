DO
$$
DECLARE
    tabname text;
BEGIN
    -- Loop through all tables in the public schema
    FOR tabname IN
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
    LOOP
        -- Drop each table with cascade to handle dependencies
        EXECUTE format('DROP TABLE IF EXISTS public.%I CASCADE;', tabname);
    END LOOP;
END;
$$;