// Typeset the `\(...\)` and `\[...\]` spans arithmatex leaves in the page. Subscribed to
// `document$` so pages reached by instant navigation are typeset too.
document$.subscribe(({ body }) => {
  renderMathInElement(body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
  });
});
