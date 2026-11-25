-- Abnormal returns around each news event.
--
-- Inputs:  events(id, ticker, label, label_source, published_at)
--          daily_prices(date, ticker, close)  -- adjusted close
-- Params:  $index_symbol  index whose trading dates form the calendar
--          $market_close  local time; news after it counts from the next day
--          $windows       list of k
--
-- For each event and k: return from the close of session -1 to the close of
-- session k-1, for the stock and the index; abnormal = stock - index.
-- A window is NULL when any price it needs is missing.

WITH calendar AS (
    -- trading sessions = dates with an index price, numbered 1, 2, ...
    SELECT date, row_number() OVER (ORDER BY date) AS session
    FROM daily_prices
    WHERE ticker = $index_symbol
),

first_price AS (
    SELECT ticker, min(date) AS first_date
    FROM daily_prices
    GROUP BY ticker
),

news AS (
    -- published_at is 'YYYY-MM-DDTHH:MM:SS+07:00'; keep local date and time
    SELECT
        id, ticker, label, label_source, published_at,
        CAST(left(published_at, 19) AS TIMESTAMP) AS local_ts
    FROM events
),

news_day AS (
    SELECT
        *,
        CAST(local_ts AS DATE)
        + CASE WHEN CAST(local_ts AS TIME) > CAST($market_close AS TIME)
               THEN 1 ELSE 0 END AS ref_date
    FROM news
),

listed AS (
    -- drop news published before the stock has any price (pre-listing)
    SELECT n.*
    FROM news_day AS n
    JOIN first_price AS f USING (ticker)
    WHERE n.ref_date >= f.first_date
),

event_session AS (
    -- session 0 = first trading session on or after ref_date
    SELECT l.*, c.session AS session0, c.date AS event_date
    FROM listed AS l
    ASOF LEFT JOIN calendar AS c ON l.ref_date <= c.date
),

spans AS (
    SELECT
        e.*,
        w.k,
        c_start.date AS start_date,  -- session -1
        c_end.date AS end_date       -- session k-1
    FROM event_session AS e
    CROSS JOIN (SELECT unnest($windows) AS k) AS w
    LEFT JOIN calendar AS c_start ON c_start.session = e.session0 - 1
    LEFT JOIN calendar AS c_end ON c_end.session = e.session0 + w.k - 1
)

SELECT
    s.id, s.ticker, s.label, s.label_source, s.published_at,
    s.event_date, s.k, s.start_date, s.end_date,
    ps.close AS stock_start,
    pe.close AS stock_end,
    xs.close AS index_start,
    xe.close AS index_end,
    pe.close / ps.close - 1 AS stock_return,
    xe.close / xs.close - 1 AS index_return,
    (pe.close / ps.close) - (xe.close / xs.close) AS abnormal_return
FROM spans AS s
LEFT JOIN daily_prices AS ps ON ps.ticker = s.ticker AND ps.date = s.start_date
LEFT JOIN daily_prices AS pe ON pe.ticker = s.ticker AND pe.date = s.end_date
LEFT JOIN daily_prices AS xs
    ON xs.ticker = $index_symbol AND xs.date = s.start_date
LEFT JOIN daily_prices AS xe
    ON xe.ticker = $index_symbol AND xe.date = s.end_date
ORDER BY s.id, s.k
