# JSON Stroke Schema

## Envelope
* `version: 1` (optional but recommended).
* `intent`: `"complete" | "hint" | "import"` (anything is fine for import).
* `strokes`: array of stroke objects (`AIStrokeV11`); you may also wrap them under `payload.strokes` or just provide the array directly—the importer handles all three cases.
## Stroke (AIStrokeV11) Fields

* `id` (string, required): unique per stroke.
* `tool` (string, required): one of
  * `pen` – freehand polyline
  * `line` – straight segment
  * `polyline` / poly – multi-segment shapes
  * `rect` – axis-aligned rectangle
  * `ellipse` – ellipse/oval
  * `bezier` – smooth freehand (treated like pen)
  * `eraser` – eraser path
  * `text` – text box (special handling)
  * any other string is accepted but treated like `pen`.
* `points` (array, required except `text` with bbox): each item `[x, * y, t?, pressure?]`.
  * `pen/polyline/poly/bezier`: full path; coordinates in canvas * space.
  * `line`: only first & last point are used.
  * `rect/ellipse`: two opposite corners `[x0,y0]`, `[x1,y1]`.
  * `text`: two corners of the bounding box; if omitted, importer * falls back to `{x,y,w,h}`.
* `style` (object, optional): `{ size: "s"|"m"|"l"|"xl", color: * ColorName, opacity: number }`.
* `meta` (object, optional): free-form; importer preserves it.
  * For `text`, include:
    * `text`: string content (required for render)
    * `summary`: optional
    * `fontFamily`, `fontSize`, `fontWeight`, `growDir`, `lineHeight`, `padding`, `configuredWidth`, `configuredHeight` as needed.

## Tool-Specific Notes

`pen / bezier`

```
{
  "id": "stroke-001",
  "tool": "pen",
  "points": [[120, 180, 0, 0.5], [150, 210, 0.1, 0.48], [190, 240, 0.2, 0.52]],
  "style": { "size": "m", "color": "black", "opacity": 1 },
  "meta": { "author": "user" }
}
```

`line`

```
{ "id": "stroke-010", "tool": "line", "points": [[20, 40], [220, 140]] }
```

`polyline / poly`

Same shape as pen, just indicates polygonal segments.

`rect`
```
{
  "id": "stroke-020",
  "tool": "rect",
  "points": [[100, 80], [260, 160]],
  "style": { "size": "l", "color": "light-blue" }
}
```
`ellipse`
```
{ "id": "stroke-030", "tool": "ellipse", "points": [[300, 200], [420, 320]] }
```
`eraser`

Path behaves like pen; importer creates erase gestures.
```
{ "id": "stroke-040", "tool": "eraser", "points": [[150, 150], [155, 180], [160, 210]] }
```
`text`
```
{
  "id": "stroke-050",
  "tool": "text",
  "points": [[500, 120], [740, 260]],
  "style": { "size": "m", "color": "black" },
  "meta": {
    "text": "Optics Overview",
    "summary": "Core optical concepts",
    "fontFamily": "Inter",
    "fontSize": 22,
    "fontWeight": "600",
    "growDir": "down",
    "lineHeight": 1.4,
    "padding": 8
  }
}
```
## Full File Example
```
{
  "version": 1,
  "intent": "import",
  "strokes": [
    { ...pen stroke... },
    { ...rect stroke... },
    { ...text stroke... }
  ]
}
```
You can also save/export as:
```
{
  "shapes": [
    { "id": "shape-1", "kind": "pen", "x": 0, "y": 0, "points": [...] },
    { "id": "shape-2", "kind": "text", "x": 500, "y": 120, "w": 240, "h": 140, "text": "...", "meta": {...} }
  ]
}
```
The importer converts shapes into AIStrokeV11 automatically, so either representation works.