-- Enable required extensions
create extension if not exists vector;
create extension if not exists "uuid-ossp";

-- ── Projects ─────────────────────────────────────────────────────────────────
create table if not exists projects (
    project_id  uuid primary key default uuid_generate_v4(),
    name        text unique not null,
    created_at  timestamptz default now()
);

-- ── Papers ───────────────────────────────────────────────────────────────────
do $$ begin
    create type inclusion_status as enum ('pending', 'included', 'excluded');
exception when duplicate_object then null;
end $$;

create table if not exists papers (
    paper_id          uuid primary key default uuid_generate_v4(),
    project_id        uuid not null references projects(project_id) on delete cascade,

    -- External identifiers
    doi               text,
    arxiv_id          text,
    s2_id             text,
    openalex_id       text,

    -- Core metadata
    title             text not null,
    abstract          text,
    year              int,
    venue             text,
    authors           jsonb,          -- [{name, s2_id, openalex_id}]

    -- Citation counts (from S2 citationCount / OpenAlex cited_by_count)
    citation_count    int,            -- incoming: how many papers cite this one
    reference_count   int,            -- outgoing: how many papers this one cites (survey signal)

    -- SPECTER2 embedding from Semantic Scholar (dim=768)
    embedding         vector(768),

    -- Screening
    inclusion_status  inclusion_status not null default 'pending',
    screening_round   int,
    criteria_version  int,
    rejection_reason  text,

    -- Provenance
    source            text,           -- 'seed' | 'citation' | 'search'
    depth             int default 0,  -- 0 = seed, 1 = direct citation, 2 = depth-2

    created_at        timestamptz default now()
);

-- Unique constraints for upsert conflict targeting (NULLs are distinct in Postgres)
alter table papers add constraint papers_project_s2_uniq unique (project_id, s2_id);
alter table papers add constraint papers_project_doi_uniq unique (project_id, doi);

-- ANN index for cosine similarity search
create index if not exists papers_embedding_idx
    on papers using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- ── Citations ────────────────────────────────────────────────────────────────
create table if not exists citations (
    id               bigserial primary key,
    project_id       uuid not null references projects(project_id) on delete cascade,
    citing_paper_id  uuid not null references papers(paper_id) on delete cascade,
    cited_paper_id   uuid not null references papers(paper_id) on delete cascade,
    unique (citing_paper_id, cited_paper_id)
);

-- ── Criteria ─────────────────────────────────────────────────────────────────
create table if not exists criteria (
    id          bigserial primary key,
    project_id  uuid not null references projects(project_id) on delete cascade,
    version     int not null,
    content     text not null,
    trigger     text,               -- reason this version was created
    created_at  timestamptz default now(),
    unique (project_id, version)
);

-- ── Iterations ───────────────────────────────────────────────────────────────
create table if not exists iterations (
    id                       bigserial primary key,
    project_id               uuid not null references projects(project_id) on delete cascade,
    round                    int not null,
    query                    text,
    yield_rate               float,   -- % new candidates included
    overlap_rate             float,   -- % new candidates already in corpus
    mean_embedding_distance  float,   -- mean cosine dist to corpus centroid
    criteria_version         int,
    created_at               timestamptz default now()
);

-- ── Extractions (post-stability) ─────────────────────────────────────────────
create table if not exists extractions (
    id          bigserial primary key,
    paper_id    uuid not null references papers(paper_id) on delete cascade,
    project_id  uuid not null references projects(project_id) on delete cascade,
    data        jsonb not null,   -- {methods, setting, findings, limitations, related_methods}
    created_at  timestamptz default now(),
    unique (paper_id, project_id)
);
