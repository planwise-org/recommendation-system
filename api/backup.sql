--
-- PostgreSQL database dump
--

-- Dumped from database version 15.12 (Debian 15.12-1.pgdg120+1)
-- Dumped by pg_dump version 15.12 (Debian 15.12-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: placetype; Type: TYPE; Schema: public; Owner: user
--

CREATE TYPE public.placetype AS ENUM (
    'RESTAURANT',
    'CAFE',
    'BAR',
    'CLUB',
    'SHOPPING',
    'ATTRACTION',
    'PARK',
    'MUSEUM',
    'THEATER',
    'OTHER'
);


ALTER TYPE public.placetype OWNER TO "user";

--
-- Name: recommendationalgorithm; Type: TYPE; Schema: public; Owner: user
--

CREATE TYPE public.recommendationalgorithm AS ENUM (
    'AUTOENCODER',
    'SVD',
    'TRANSFER_LEARNING'
);


ALTER TYPE public.recommendationalgorithm OWNER TO "user";

--
-- Name: userrole; Type: TYPE; Schema: public; Owner: user
--

CREATE TYPE public.userrole AS ENUM (
    'ADMIN',
    'USER'
);


ALTER TYPE public.userrole OWNER TO "user";

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: place; Type: TABLE; Schema: public; Owner: user
--

CREATE TABLE public.place (
    id integer NOT NULL,
    name character varying NOT NULL,
    description character varying NOT NULL,
    address character varying NOT NULL,
    latitude double precision NOT NULL,
    longitude double precision NOT NULL,
    place_type public.placetype NOT NULL,
    rating double precision NOT NULL,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);


ALTER TABLE public.place OWNER TO "user";

--
-- Name: place_id_seq; Type: SEQUENCE; Schema: public; Owner: user
--

CREATE SEQUENCE public.place_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.place_id_seq OWNER TO "user";

--
-- Name: place_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: user
--

ALTER SEQUENCE public.place_id_seq OWNED BY public.place.id;


--
-- Name: recommendation; Type: TABLE; Schema: public; Owner: user
--

CREATE TABLE public.recommendation (
    id integer NOT NULL,
    user_id integer NOT NULL,
    place_id integer NOT NULL,
    algorithm public.recommendationalgorithm NOT NULL,
    score double precision NOT NULL,
    visited boolean NOT NULL,
    reviewed boolean NOT NULL,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);


ALTER TABLE public.recommendation OWNER TO "user";

--
-- Name: recommendation_id_seq; Type: SEQUENCE; Schema: public; Owner: user
--

CREATE SEQUENCE public.recommendation_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.recommendation_id_seq OWNER TO "user";

--
-- Name: recommendation_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: user
--

ALTER SEQUENCE public.recommendation_id_seq OWNED BY public.recommendation.id;


--
-- Name: review; Type: TABLE; Schema: public; Owner: user
--

CREATE TABLE public.review (
    id integer NOT NULL,
    user_id integer NOT NULL,
    place_id integer NOT NULL,
    rating double precision NOT NULL,
    comment character varying NOT NULL,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);


ALTER TABLE public.review OWNER TO "user";

--
-- Name: review_id_seq; Type: SEQUENCE; Schema: public; Owner: user
--

CREATE SEQUENCE public.review_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.review_id_seq OWNER TO "user";

--
-- Name: review_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: user
--

ALTER SEQUENCE public.review_id_seq OWNED BY public.review.id;


--
-- Name: user; Type: TABLE; Schema: public; Owner: user
--

CREATE TABLE public."user" (
    id integer NOT NULL,
    name character varying NOT NULL,
    email character varying NOT NULL,
    hash_password character varying NOT NULL,
    preferences character varying NOT NULL
);


ALTER TABLE public."user" OWNER TO "user";

--
-- Name: user_id_seq; Type: SEQUENCE; Schema: public; Owner: user
--

CREATE SEQUENCE public.user_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.user_id_seq OWNER TO "user";

--
-- Name: user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: user
--

ALTER SEQUENCE public.user_id_seq OWNED BY public."user".id;


--
-- Name: place id; Type: DEFAULT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.place ALTER COLUMN id SET DEFAULT nextval('public.place_id_seq'::regclass);


--
-- Name: recommendation id; Type: DEFAULT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.recommendation ALTER COLUMN id SET DEFAULT nextval('public.recommendation_id_seq'::regclass);


--
-- Name: review id; Type: DEFAULT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.review ALTER COLUMN id SET DEFAULT nextval('public.review_id_seq'::regclass);


--
-- Name: user id; Type: DEFAULT; Schema: public; Owner: user
--

ALTER TABLE ONLY public."user" ALTER COLUMN id SET DEFAULT nextval('public.user_id_seq'::regclass);


--
-- Data for Name: place; Type: TABLE DATA; Schema: public; Owner: user
--

COPY public.place (id, name, description, address, latitude, longitude, place_type, rating, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: recommendation; Type: TABLE DATA; Schema: public; Owner: user
--

COPY public.recommendation (id, user_id, place_id, algorithm, score, visited, reviewed, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: review; Type: TABLE DATA; Schema: public; Owner: user
--

COPY public.review (id, user_id, place_id, rating, comment, created_at, updated_at) FROM stdin;
\.


--
-- Data for Name: user; Type: TABLE DATA; Schema: public; Owner: user
--

COPY public."user" (id, name, email, hash_password, preferences) FROM stdin;
\.


--
-- Name: place_id_seq; Type: SEQUENCE SET; Schema: public; Owner: user
--

SELECT pg_catalog.setval('public.place_id_seq', 1, false);


--
-- Name: recommendation_id_seq; Type: SEQUENCE SET; Schema: public; Owner: user
--

SELECT pg_catalog.setval('public.recommendation_id_seq', 1, false);


--
-- Name: review_id_seq; Type: SEQUENCE SET; Schema: public; Owner: user
--

SELECT pg_catalog.setval('public.review_id_seq', 1, false);


--
-- Name: user_id_seq; Type: SEQUENCE SET; Schema: public; Owner: user
--

SELECT pg_catalog.setval('public.user_id_seq', 1, false);


--
-- Name: place place_pkey; Type: CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.place
    ADD CONSTRAINT place_pkey PRIMARY KEY (id);


--
-- Name: recommendation recommendation_pkey; Type: CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.recommendation
    ADD CONSTRAINT recommendation_pkey PRIMARY KEY (id);


--
-- Name: review review_pkey; Type: CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.review
    ADD CONSTRAINT review_pkey PRIMARY KEY (id);


--
-- Name: user user_pkey; Type: CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public."user"
    ADD CONSTRAINT user_pkey PRIMARY KEY (id);


--
-- Name: ix_user_name; Type: INDEX; Schema: public; Owner: user
--

CREATE INDEX ix_user_name ON public."user" USING btree (name);


--
-- Name: recommendation recommendation_place_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.recommendation
    ADD CONSTRAINT recommendation_place_id_fkey FOREIGN KEY (place_id) REFERENCES public.place(id);


--
-- Name: recommendation recommendation_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.recommendation
    ADD CONSTRAINT recommendation_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);


--
-- Name: review review_place_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.review
    ADD CONSTRAINT review_place_id_fkey FOREIGN KEY (place_id) REFERENCES public.place(id);


--
-- Name: review review_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.review
    ADD CONSTRAINT review_user_id_fkey FOREIGN KEY (user_id) REFERENCES public."user"(id);


--
-- PostgreSQL database dump complete
--

