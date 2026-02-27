// 1) Papers citing a given paper
MATCH (citing:Paper)-[:CITES]->(target:Paper {bibcode: $bibcode})
RETURN citing.bibcode AS bibcode, citing.title AS title, citing.year AS year
ORDER BY year DESC;

// 2) Most cited papers (metadata citation_count)
MATCH (p:Paper)
RETURN p.bibcode AS bibcode, p.title AS title, p.citation_count AS citation_count
ORDER BY citation_count DESC
LIMIT 20;

// 3) Related papers by shared keywords
MATCH (seed:Paper {bibcode: $bibcode})-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(related:Paper)
WHERE related.bibcode <> seed.bibcode
RETURN related.bibcode AS bibcode, related.title AS title, count(DISTINCT k) AS shared_keywords
ORDER BY shared_keywords DESC, related.citation_count DESC
LIMIT 20;

// 4) Authors of a paper
MATCH (a:Author)-[w:WROTE]->(p:Paper {bibcode: $bibcode})
RETURN a.name AS author, w.author_order AS author_order
ORDER BY author_order ASC;

// 5) Papers by author
MATCH (a:Author {name: $author_name})-[:WROTE]->(p:Paper)
RETURN p.bibcode AS bibcode, p.title AS title, p.year AS year
ORDER BY year DESC, title ASC
LIMIT 50;

// 6) Venue distribution
MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue)
RETURN v.name AS venue, count(*) AS paper_count
ORDER BY paper_count DESC
LIMIT 25;

// 7) Coauthor pairs by shared papers
MATCH (a1:Author)-[:WROTE]->(p:Paper)<-[:WROTE]-(a2:Author)
WHERE a1.name < a2.name
RETURN a1.name AS author_1, a2.name AS author_2, count(DISTINCT p) AS shared_papers
ORDER BY shared_papers DESC
LIMIT 25;

// 8) Shortest citation path between two papers (bounded depth)
MATCH path = shortestPath((src:Paper {bibcode: $source_bibcode})-[:CITES*..6]->(dst:Paper {bibcode: $target_bibcode}))
RETURN path;
