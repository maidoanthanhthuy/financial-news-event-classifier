# Labeling Guide

Assign exactly **one label** to each news item, based on the **main event in the
title**. Use the sapo (lead paragraph) only to clarify an unclear title.

## Labels

### `business_results`
Revenue, profit or loss for a quarter or year; annual profit targets.
- "Vinamilk lãi 2.500 tỷ đồng quý III, tăng 12%" (Vinamilk earns VND 2.5 trillion in Q3, up 12%)
- "Hòa Phát đặt mục tiêu lợi nhuận 15.000 tỷ đồng năm 2026" (Hoa Phat targets VND 15 trillion profit for 2026)

### `dividends_issuance`
Cash or stock dividends, new share issuance, share buybacks (treasury shares).
- "FPT chốt quyền trả cổ tức tiền mặt 20%" (FPT sets record date for a 20% cash dividend)
- "Techcombank phát hành thêm 350 triệu cổ phiếu" (Techcombank issues 350 million new shares)

### `macro_policy`
Interest rates, exchange rates, State Bank of Vietnam (SBV) decisions, inflation,
GDP, tax policy, public investment. Taxes and tariffs set by foreign countries
(including anti-dumping duties) belong here too, even if the title names only one
company.
- "Ngân hàng Nhà nước giữ nguyên lãi suất điều hành" (SBV keeps policy rates unchanged)
- "GDP quý III tăng 7,4%" (Q3 GDP grows 7.4%)

### `ma_investment`
Mergers and acquisitions, buying stakes in other companies, new projects or plants.
- "Masan mua 30% cổ phần chuỗi cà phê Phúc Long" (Masan buys a 30% stake in Phuc Long coffee chain)
- "Vingroup khởi công nhà máy xe điện 4 tỷ USD" (Vingroup breaks ground on a USD 4 billion EV plant)

### `legal_personnel`
Fines, lawsuits, violations; changes in company leadership (chair, CEO, board).
Appointments at government agencies (ministries, SBV and its branches) are `other`.
- "Ủy ban Chứng khoán phạt công ty X 500 triệu đồng" (Securities regulator fines company X VND 500 million)
- "Vietcombank bổ nhiệm tổng giám đốc mới" (Vietcombank appoints a new CEO)

### `other`
Daily market reports, insiders or major shareholders buying/selling shares,
awards, brokerage recommendations, lifestyle news, bonds (issuance, use of proceeds,
late payments), appointments at government agencies, and anything not covered above.
- "VN-Index tăng 15 điểm, thanh khoản cải thiện" (VN-Index gains 15 points, liquidity improves)
- "Chủ tịch HĐQT đăng ký mua 5 triệu cổ phiếu" (Board chair registers to buy 5 million shares)

## Edge cases
- Several events in one title: pick the one stated first or emphasized most.
- Shareholder meeting news: label by the main topic (profit plan ->
  `business_results`, dividend plan -> `dividends_issuance`).
- Insider or major-shareholder trades are `other`, not `dividends_issuance`.
- A company buying its own shares (buyback) is `dividends_issuance`; buying shares of
  another company is `ma_investment`.
- Sector or market commentary without a specific company event is `other`, unless it
  is about a macro or policy topic (`macro_policy`).
- Bonds are always `other`, not `dividends_issuance` (which covers shares only).
- Leadership changes count as `legal_personnel` only at companies; government
  appointments are `other`.
- Taxes or foreign anti-dumping duties are `macro_policy`, even when only one company
  is mentioned.
- Forecasts or estimates by brokerages or other third parties are `other`. Plans or
  targets announced by the company itself stay `business_results`.
- News covering many companies or a whole sector is still labeled by its main
  topic. (The event study later keeps only news tied to exactly one VN30 ticker.)
- Foreign companies not listed in Vietnam are `other`.
- A stock placed under warning, control or trading restriction is `legal_personnel`.
- `legal_personnel` needs a specific company or one of its leaders (current or
  former) as the subject. Criminal cases or scams not tied to a company are `other`.
- `ma_investment` needs at least one Vietnamese company (listed or not) taking part.
- Intentions or investments only "being considered" (not yet a decided event) are
  `other`.
- Sales contracts, orders, new flight routes or new stores (day-to-day business) are
  `other`.
- Profit from selling a stake (divestment gain) is `business_results`.
- Forecasts of economic figures (GDP, exchange rate, growth) are `other`. Discussion
  of, or proposals about, a specific policy are `macro_policy`.
- Interest rates set by a single bank for its own products are `other`. Policy rates
  or SBV directions on interest rates are `macro_policy`.

## Output
Return the label and a short reason in English (under 15 words).
