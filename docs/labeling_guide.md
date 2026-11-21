# Labeling Guide

Assign exactly **one of 6 labels** to each news item, based on the **main event in
the title**. Use the sapo (lead paragraph) only to clarify an unclear title.
`other` is a real label like the other five, not a reject bin: use it whenever no
event type fits.

## Labels

### `business_results`
Revenue, profit or loss for a quarter or year; operating figures (cars sold,
output volume, passenger numbers); plans, targets or estimates of these results
announced by the company itself.
- "Vinamilk lãi 2.500 tỷ đồng quý III, tăng 12%" (Vinamilk earns VND 2.5 trillion in Q3, up 12%)
- "Hòa Phát đặt mục tiêu lợi nhuận 15.000 tỷ đồng năm 2026" (Hoa Phat targets VND 15 trillion profit for 2026)

### `dividends_issuance`
Cash or stock dividends, new share issuance (including charter capital increases),
share buybacks (treasury shares).
- "FPT chốt quyền trả cổ tức tiền mặt 20%" (FPT sets record date for a 20% cash dividend)
- "Techcombank phát hành thêm 350 triệu cổ phiếu" (Techcombank issues 350 million new shares)

### `macro_policy`
Interest rates, State Bank of Vietnam (SBV) decisions and exchange-rate policy
(central rate), inflation, GDP, credit growth, tax policy, public investment. Data
releases on credit, GDP and inflation belong here. Taxes and tariffs set by foreign
countries (including anti-dumping duties) belong here too, even if the title names
only one company.
- "Ngân hàng Nhà nước giữ nguyên lãi suất điều hành" (SBV keeps policy rates unchanged)
- "GDP quý III tăng 7,4%" (Q3 GDP grows 7.4%)

### `ma_investment`
M&A and corporate investment: mergers and acquisitions, buying shares or stakes
in other companies, new projects, plants and capacity expansion (greenfield).
- "Masan mua 30% cổ phần chuỗi cà phê Phúc Long" (Masan buys a 30% stake in Phuc Long coffee chain)
- "Vingroup khởi công nhà máy xe điện 4 tỷ USD" (Vingroup breaks ground on a USD 4 billion EV plant)

### `legal_personnel`
Fines, lawsuits, violations; changes in company leadership (chair, CEO, board).
"Personnel" means leadership changes only (board of directors, executive team);
staff cuts or layoffs do not count.
Appointments at government agencies (ministries, SBV and its branches) are `other`.
- "Ủy ban Chứng khoán phạt công ty X 500 triệu đồng" (Securities regulator fines company X VND 500 million)
- "Vietcombank bổ nhiệm tổng giám đốc mới" (Vietcombank appoints a new CEO)

### `other`
News with no event of the five types above. The model learns and predicts this
label like any other. Examples: daily market reports, insiders or major
shareholders buying/selling shares, awards, brokerage recommendations, lifestyle
news, bonds (issuance, use of proceeds, late payments), appointments at government
agencies, and anything not covered above.
- "VN-Index tăng 15 điểm, thanh khoản cải thiện" (VN-Index gains 15 points, liquidity improves)
- "Chủ tịch HĐQT đăng ký mua 5 triệu cổ phiếu" (Board chair registers to buy 5 million shares)

## Edge cases
- Several events in one title: pick the one stated first or emphasized most.
- Shareholder meeting news: label by the main topic (profit plan ->
  `business_results`, dividend plan -> `dividends_issuance`).
- Trades on the exchange by insiders, major shareholders or their related persons
  (e.g. relatives) are `other`. A company buying treasury shares or issuing shares
  in a private placement is `dividends_issuance`.
- A company buying its own shares (buyback) is `dividends_issuance`; buying shares of
  another company is `ma_investment`.
- Sector or market commentary without a specific company event is `other`, unless it
  is about a macro or policy topic (`macro_policy`).
- Bonds are always `other`, not `dividends_issuance` (which covers shares only).
- Leadership changes count as `legal_personnel` only at companies; government
  appointments are `other`.
- Taxes or foreign anti-dumping duties are `macro_policy`, even when only one company
  is mentioned.
- News covering many companies or a whole sector is still labeled by its main
  topic. (The event study later keeps only news tied to exactly one VN30 ticker.)
- Foreign companies not listed in Vietnam are `other`.
- A stock placed under warning, control or trading restriction is `legal_personnel`.
- `legal_personnel` needs a specific company or one of its leaders (current or
  former) as the subject. Criminal cases or scams not tied to a company are `other`.
- `ma_investment` needs at least one Vietnamese company (listed or not) taking part.
- Intentions or investments only "being considered" (not yet a decided event) are
  `other`.
- Sales contracts, orders, new flight routes, new stores or branches, and expanding
  an existing service network (day-to-day business) are `other`. Building a new
  plant or adding production capacity is `ma_investment`.
- Profit from selling a stake (divestment gain) is `business_results`.
- Forecasts of economic figures (GDP, exchange rate, growth) are `other`. Discussion
  of, or proposals about, a specific policy are `macro_policy`.
- Interest rates set by a single bank for its own products are `other`. Policy rates
  or SBV directions on interest rates are `macro_policy`.
- Results plans, targets or estimates: announced by the company itself ->
  `business_results`; stated by a third party -> `other`. Judge by what is known
  at the time the news was published.
- A charter capital increase is share issuance -> `dividends_issuance`. Only
  comparing or ranking companies by capital size is `other`.
- Look-back pieces and general articles with no new event are `other`.
- Daily USD prices (bank or black-market rates) are `other`. The SBV central
  exchange rate or SBV exchange-rate policy is `macro_policy`.
- Listing changes: moving to HOSE voluntarily or becoming a public company ->
  `dividends_issuance`; forced move to a lower exchange or delisting because of
  violations -> `legal_personnel`; deregistration from trading with no stated
  reason -> `other`.
- Cooperation agreements, MOUs or pledges to "support" without a capital
  contribution are `other`. Contributing capital to a joint venture or buying
  shares is `ma_investment`.
- An exchange launching new indices or rebalancing index baskets is `other`.

## Output
Return the label and a short reason in English (under 15 words).
