# Frontend redesign — critique and proposal

Status: approved (Style A, calm clinical) and implemented. The critique below describes the previous design; `outputs/figures/app/before/` keeps screenshots of it.

Mock-ups (static, synthetic signals, illustrative numbers): open `mockup.html` in a browser with
`?style=a|b&page=prediction|comparison`, or look at the four PNG files next to it.

| | Prediction | Comparison |
|---|---|---|
| Style A — calm clinical (light) | `mockup_a_prediction.png` | `mockup_a_comparison.png` |
| Style B — dark console | `mockup_b_prediction.png` | `mockup_b_comparison.png` |

## 1. Critique of the current app

Measured on a 1920×1080 screen, in priority order.

1. **The EEG is not the hero.** On Prediction the plot starts 393 px down the page and is 1,192 px tall. Only 13 of the 19 channels are visible; the probability row, the threshold line and the navigator are off-screen. You must scroll to see what the model predicts.
2. **Three stacked panels sit above the plot** (patient facts, verdict, "this model on all patients"). They take 280 px and repeat information: the patient is already named in the sidebar, the model in the header.
3. **The channel selector is a wall of 19 pills** with an × each. It takes 220 px of the sidebar, does not show the scalp regions it is grouped by, and makes "hide the temporal channels" a four-click task.
4. **No clear hierarchy.** Section labels, card titles, values and captions use similar sizes and weights; six different text styles appear in the first screen. Nothing says "look here first".
5. **Too many competing colours at once.** Teal, blue, pink, sky, green, coral, amber, violet, orange and three badge colours all appear on one screen. The accent (teal) is also the frontal-channel colour and close to the seizure green.
6. **Comparison has no narrative.** It opens with selection controls and four equal cards; the winner is not called out, and the reader must find it in a 16-column table that scrolls sideways.
7. **Two legends say different things.** The sidebar legend ("Technician event") and the plot legend ("Technician note") name the same item differently, and the sidebar one takes space on every page, including Comparison where half of it does not apply.
8. **Inconsistent components.** Card paddings differ (12, 16, 24 px), badges are uppercase in some places and not in others, the plot toolbar mixes outlined buttons with Plotly's own icon bar, and tooltips are large enough to cover neighbouring controls.
9. **The header is crowded.** Brand, subtitle, three page tabs, two segmented controls, a badge repeating their value, and the theme button compete in one 60 px strip; the model controls show on Visualisation, where no model is used.
10. **States are undesigned.** Loading is a default dot spinner, the empty spectrum is a large blank card, and a patient with no detections shows an empty table with no message.

## 2. Proposal

### Structure (same in both styles)

- **Header, 52 px**: brand, the three parts as tabs, and one labelled *Model* group (algorithm + training mode) that is hidden on Visualisation. The badge that repeated the model goes away.
- **Sidebar, 272 px**: *Patient* (selector plus a compact facts box: sex, age, duration, real seizures, annotation source), *Threshold* (Prediction only), *Channels* as 19 toggle chips in five region rows with a "19 of 19" counter and "Show all", *View* (stacked / overlay, page length, gain), then the two switches. The sidebar legend is removed.
- **Prediction / Visualisation main area**:
  - one 64 px summary strip: verdict, one sentence, and on the right the model's all-patient scores;
  - the EEG card fills the rest of the screen. Row height adapts to the window so that 19 channels, the probability row and the navigator fit in 1080 px without scrolling. Its header holds paging, seizure jumps, the time range and the exports; its footer holds the one legend and the interaction hint;
  - *Spectrum* and *Intervals* open on demand (buttons in the plot header) instead of always occupying space below.
- **Comparison, read as a story**:
  1. *Who wins*: the best model in a larger, outlined card with its numbers and a one-sentence reason; the others in smaller cards, in rank order.
  2. *Why*: event F1 against the threshold, precision–recall, and the training class distribution side by side.
  3. *Detail*: ranking table (fits without sideways scrolling at 1920 px; best values in the accent colour), then confusion matrices, the patient timeline and the per-patient table, unchanged in content.

### Visual system

- **Typography**: Inter for text, JetBrains Mono (tabular digits) for every number, time and channel name. Four levels only: section label (11 px, uppercase, spaced), card title (15 px, 600), body (13–14 px), value (15 px mono; 44 px for the hero number).
- **Spacing**: 4 / 8 px grid; every card has 10 px radius, a 1 px border, 16 px padding and the same faint shadow.
- **One accent colour** for selection, focus and the "best" highlight. Meaning colours keep their roles everywhere: green = real seizure, coral = detection, amber = technician note, violet = threshold, one colour per algorithm, solid = unbalanced, dashed = balanced. Each is always paired with a label or legend.
- **Plots** take their fonts, grid and colours from the same tokens as the page and follow the light/dark toggle.
- **States**: skeleton placeholders while loading; short centred messages for an empty selection, missing data, and "no detection at this threshold".

### The two styles

| | Style A — calm clinical | Style B — dark console |
|---|---|---|
| Default theme | Light (dark stays available through the toggle) | Dark (light stays available) |
| Background / surface | `#F5F7FA` / white | `#0A0F1C` / `#101827` |
| Accent | Deep teal `#0B7285` | Cyan `#4FD1C5` |
| Character | Medical-device console; best for report screenshots and printing | Monitoring room; traces glow against the background |
| Risk | Thin coloured traces need slightly darker shades to stay readable on white | Less suited to printed reports; needs care with contrast of small grey text |

Both use the same layout and components; the choice is which one is the default and gets the most polish. The other remains reachable with the toggle.

## 3. What will not change

- Every control, every number, every callback id and all model and data code.
- The three parts and what each shows.
- English labels.

## 4. Open points to decide

1. **Style A or B as the default.**
2. **Spectrum and Intervals on demand** (hidden until asked for) or always visible below the plot as today.
3. **Channel chips** instead of the pill list. The id and behaviour stay the same; only the look changes.
4. **Model controls hidden on Visualisation**, where they have no effect.
