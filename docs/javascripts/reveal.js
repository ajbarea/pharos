// Landing page behaviour: reveal-on-scroll, and the particle field behind the hero.
//
// Progressive enhancement in both directions. The CSS keeps every section visible
// until this file sets `js-ready`, so a reader without JavaScript sees the whole
// page rather than a blank one; and a reader who has asked for reduced motion gets
// every section marked visible at once and no canvas at all.
//
// Re-entrant, because Material's instant navigation swaps the document body without
// a page load: a one-shot listener would bind on first visit and never again. Every
// binding this file makes is torn down before the next one is created.
(function () {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  document.documentElement.classList.add("js-ready");

  let teardown = null;

  // Motes drift up through the beams. Colours are the three lattice axes, and they
  // are read from CSS rather than duplicated here so the light-theme overrides in
  // landing.css apply to the canvas too.
  function beamColours() {
    const style = getComputedStyle(document.documentElement);
    return ["--pharos-amber", "--pharos-cyan", "--pharos-magenta"]
      .map((name) => style.getPropertyValue(name).trim())
      .filter(Boolean);
  }

  function startParticles(hero) {
    const canvas = document.createElement("canvas");
    canvas.className = "hero-particles";
    canvas.setAttribute("aria-hidden", "true");
    hero.insertBefore(canvas, hero.firstChild);

    const ctx = canvas.getContext("2d");
    let colours = beamColours();
    let particles = [];
    let animId = null;
    let width = 0;
    let height = 0;

    // Back the canvas at device resolution. Without this the motes are visibly
    // soft on any HiDPI screen, which is most of them.
    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = hero.offsetWidth;
      height = hero.offsetHeight;
      canvas.width = Math.max(1, Math.floor(width * dpr));
      canvas.height = Math.max(1, Math.floor(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function spawn(atBottom) {
      return {
        x: Math.random() * width,
        y: atBottom ? height + Math.random() * 40 : Math.random() * height,
        size: Math.random() * 1.9 + 0.4,
        driftX: (Math.random() - 0.5) * 0.16,
        driftY: -(Math.random() * 0.28 + 0.06),
        colour: colours[Math.floor(Math.random() * colours.length)],
        alpha: Math.random() * 0.5 + 0.12,
        twinkleRate: Math.random() * 0.02 + 0.004,
        phase: Math.random() * Math.PI * 2,
        age: 0,
      };
    }

    function seed() {
      resize();
      // Density by area, capped so a very wide window does not draw thousands.
      const count = Math.min(140, Math.floor((width * height) / 9000));
      particles = Array.from({ length: count }, () => spawn(false));
    }

    function frame() {
      ctx.clearRect(0, 0, width, height);
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        p.x += p.driftX;
        p.y += p.driftY;
        p.age++;

        const twinkle = 0.55 + 0.45 * Math.sin(p.age * p.twinkleRate + p.phase);
        ctx.globalAlpha = p.alpha * twinkle;
        ctx.fillStyle = p.colour;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        // A soft halo on the larger motes only; on every mote it turns to fog.
        if (p.size > 1.3) {
          ctx.globalAlpha = p.alpha * twinkle * 0.16;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 3.4, 0, Math.PI * 2);
          ctx.fill();
        }

        if (p.y < -12 || p.x < -12 || p.x > width + 12) {
          particles[i] = spawn(true);
        }
      }
      ctx.globalAlpha = 1;
      animId = requestAnimationFrame(frame);
    }

    function play() {
      if (!animId) frame();
    }
    function pause() {
      if (animId) {
        cancelAnimationFrame(animId);
        animId = null;
      }
    }

    seed();
    play();

    // Do not animate a hero that has been scrolled past, or a background tab.
    const onScreen = new IntersectionObserver(
      (entries) => (entries[0].isIntersecting ? play() : pause()),
      { threshold: 0 },
    );
    onScreen.observe(hero);

    const onVisibility = () => (document.hidden ? pause() : play());
    document.addEventListener("visibilitychange", onVisibility);

    const onResize = () => seed();
    window.addEventListener("resize", onResize);

    // The palette changes when the reader flips the theme toggle; recolour rather
    // than keep drawing dark-theme motes on a white page.
    const onTheme = new MutationObserver(() => {
      colours = beamColours();
      particles.forEach((p) => {
        p.colour = colours[Math.floor(Math.random() * colours.length)];
      });
    });
    onTheme.observe(document.body, {
      attributes: true,
      attributeFilter: ["data-md-color-scheme"],
    });

    return () => {
      pause();
      onScreen.disconnect();
      onTheme.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("resize", onResize);
      canvas.remove();
    };
  }

  function init() {
    if (teardown) {
      teardown();
      teardown = null;
    }

    const sections = document.querySelectorAll(".landing-section");
    if (!sections.length) return;

    if (reduceMotion) {
      sections.forEach((s) => s.classList.add("visible"));
      return;
    }

    const revealer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("visible");
          revealer.unobserve(entry.target);
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" },
    );
    sections.forEach((s) => revealer.observe(s));

    const hero = document.querySelector(".hero");
    const stopParticles = hero ? startParticles(hero) : null;

    teardown = () => {
      revealer.disconnect();
      if (stopParticles) stopParticles();
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  }
})();
