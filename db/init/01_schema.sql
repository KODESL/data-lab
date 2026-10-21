-- Serving database schemas. Tables are (re)created by `madrid publish` from the gold
-- Parquet files; this file only prepares schemas, extensions and a couple of views.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

-- Convenience view used by BI tools once `madrid publish` has run.
CREATE OR REPLACE FUNCTION gold.ensure_views() RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  IF to_regclass('gold.lsi') IS NOT NULL THEN
    EXECUTE $v$
      CREATE OR REPLACE VIEW gold.v_lsi_latest AS
      SELECT l.*, m.name, m.is_small, m.population_latest
      FROM gold.lsi l JOIN silver.dim_municipio m USING (municipality_id)
      WHERE l.year = (SELECT max(year) FROM gold.lsi)
    $v$;
  END IF;
END $$;
