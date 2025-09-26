SET search_path TO public;


CREATE TABLE compounds (
    cid INTEGER PRIMARY KEY,
    name VARCHAR(1000) NOT NULL,
    mf VARCHAR(100) NOT NULL,
    mw REAL NOT NULL,
    charge SMALLINT NOT NULL,
    smiles VARCHAR(1000) NOT NULL,
    inchi VARCHAR(1000) NOT NULL,
    inchikey CHAR(27) UNIQUE NOT NULL,
    complexity REAL NOT NULL,
    bertz_complexity REAL NOT NULL,
    organic BOOLEAN NOT NULL
);


CREATE TABLE compound_fingerprints (
    cid INTEGER PRIMARY KEY REFERENCES compounds(cid),
    ECFP4_fp INTEGER[64] NOT NULL,
    popcount SMALLINT NOT NULL
);


CREATE TABLE compound_synonyms (
    cid INTEGER REFERENCES compounds(cid),
    synonym VARCHAR(1000),
    PRIMARY KEY (cid, synonym)
);


CREATE TABLE compound_wiki (
    cid INTEGER PRIMARY KEY REFERENCES compounds(cid),
    wiki VARCHAR(300) NOT NULL
);


CREATE TABLE compound_nfpa (
    cid INTEGER PRIMARY KEY REFERENCES compounds(cid),
    health SMALLINT,
    flammability SMALLINT,
    instability SMALLINT
);


CREATE TABLE compound_hazard_statements (
    cid INTEGER REFERENCES compounds(cid),
    statement CHAR(4) NOT NULL,
    PRIMARY KEY (cid, statement)
);


CREATE TABLE compound_hazard_pictograms (
    cid INTEGER REFERENCES compounds(cid),
    pictogram CHAR(5) NOT NULL,
    PRIMARY KEY (cid, pictogram)
);


CREATE TABLE reactions (
    rid CHAR(24) PRIMARY KEY,
    complexity REAL NOT NULL,
    source VARCHAR(100) NOT NULL,
    balanced BOOLEAN NOT NULL,
    confidence REAL
);


CREATE TABLE reaction_reactants (
    cid INTEGER NOT NULL REFERENCES compounds(cid),
    rid CHAR(24) NOT NULL REFERENCES reactions(rid),
    PRIMARY KEY (cid, rid)
);


CREATE TABLE reaction_products (
    cid INTEGER NOT NULL REFERENCES compounds(cid),
    rid CHAR(24) NOT NULL REFERENCES reactions(rid),
    PRIMARY KEY (cid, rid)
);


CREATE TABLE reaction_solvents (
    cid INTEGER NOT NULL REFERENCES compounds(cid),
    rid CHAR(24) NOT NULL REFERENCES reactions(rid),
    PRIMARY KEY (cid, rid)
);


CREATE TABLE reaction_catalysts (
    cid INTEGER NOT NULL REFERENCES compounds(cid),
    rid CHAR(24) NOT NULL REFERENCES reactions(rid),
    PRIMARY KEY (cid, rid)
);


CREATE TABLE reaction_details (
    rid CHAR(24) PRIMARY KEY REFERENCES reactions(rid),
    doi VARCHAR(100),
    patent VARCHAR(100),
    description TEXT
);