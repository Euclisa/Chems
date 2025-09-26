DO
$$
DECLARE
    tabname text;
BEGIN
    FOR tabname IN
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
    LOOP
        EXECUTE format('TRUNCATE TABLE %I.%I RESTART IDENTITY CASCADE;', 'public', tabname);
    END LOOP;
END;
$$;