// Reveal-on-scroll for the landing page.
//
// Progressive enhancement in both directions. The CSS keeps every section visible
// until this file sets `js-ready`, so a reader without JavaScript sees the whole
// page rather than a blank one; and a reader who has asked for reduced motion gets
// every section marked visible immediately rather than a subtler animation.
//
// Re-entrant, because Material's instant navigation swaps the document body without
// a page load: a one-shot listener would bind on first visit and never again.
(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.documentElement.classList.add("js-ready");

  let observer = null;

  function init() {
    if (observer) {
      observer.disconnect();
      observer = null;
    }

    const sections = document.querySelectorAll(".landing-section");
    if (!sections.length) return;

    if (reduceMotion) {
      sections.forEach((s) => s.classList.add("visible"));
      return;
    }

    observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" },
    );

    // A section already in view on load must not wait for a scroll that may never
    // come; the observer fires for those on its first callback, which covers it.
    sections.forEach((s) => observer.observe(s));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Material exposes an instant-navigation hook when the feature is enabled.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  }
})();
