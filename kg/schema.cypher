CREATE CONSTRAINT paper_bibcode_unique IF NOT EXISTS
FOR (p:Paper)
REQUIRE p.bibcode IS UNIQUE;

CREATE CONSTRAINT author_name_unique IF NOT EXISTS
FOR (a:Author)
REQUIRE a.name IS UNIQUE;

CREATE CONSTRAINT keyword_name_unique IF NOT EXISTS
FOR (k:Keyword)
REQUIRE k.name IS UNIQUE;

CREATE CONSTRAINT venue_name_unique IF NOT EXISTS
FOR (v:Venue)
REQUIRE v.name IS UNIQUE;

CREATE INDEX paper_year_idx IF NOT EXISTS
FOR (p:Paper)
ON (p.year);

CREATE INDEX paper_citation_count_idx IF NOT EXISTS
FOR (p:Paper)
ON (p.citation_count);

CREATE FULLTEXT INDEX paper_text_idx IF NOT EXISTS
FOR (p:Paper)
ON EACH [p.title, p.abstract];
