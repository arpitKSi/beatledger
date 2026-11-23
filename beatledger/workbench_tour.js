export default function mountTour() {
  if (document.getElementById("bl-tour")) return;

  function visible(selector, root = document) {
    return Array.from(root.querySelectorAll(selector)).find(element =>
      element.getClientRects().length && !element.closest("[hidden], [inert]")
    );
  }

  const tab = name => Array.from(document.querySelectorAll('[role="tab"]'))
    .find(element => element.textContent.trim() === name);
  const widget = key => visible(".st-key-" + key);
  const disclosure = name => Array.from(document.querySelectorAll("details > summary"))
    .find(element => element.getClientRects().length &&
      Array.from(element.querySelectorAll("p")).some(label => label.textContent.trim() === name));
  const reviewPanel = () => visible('[role="tabpanel"]') || document;

  const steps = [
    {
      title: "Choose your screenplays", sidebar: true,
      target: () => widget("input_source"),
      copy: "The bundled sample is ready to explore. Upload files lets you run BeatLedger on your own timestamped .txt screenplays.",
    },
    {
      title: "Choose how cuts are scored", sidebar: true,
      target: () => widget("judge_provider") || widget("enable_judge"),
      copy: "BeatLedger uses validated model judgments and keeps baseline scores for cuts without one. Hosted models use a key you save for this session.",
    },
    {
      title: "Run the analysis", sidebar: true,
      target: () => widget("run_analysis"),
      copy: "Run analysis builds and ranks a shortlist from the selected scripts. On the sample, Open saved model run shows model scores right away.",
    },
    {
      title: "Read the ranked shortlist", tab: "Review cuts",
      target: () => visible('[data-testid="stDataFrame"]', reviewPanel()),
      copy: "Start with a high-ranked cut. The table shows each cut’s episode, duration, score and review status.",
    },
    {
      title: "Filter the shortlist", tab: "Review cuts",
      target: () => visible('[class*="st-key-archetype-"]', reviewPanel())
        ?.closest('[data-testid="stHorizontalBlock"]'),
      copy: "Hook archetype is the opening type. Tension delta measures how much the stakes change. Dialogue WPS is words per second. Review status lets you focus on cuts you still need to review.",
    },
    {
      title: "Pick a cut to inspect", tab: "Review cuts", above: true,
      target: () => visible('[class*="st-key-inspect-"]', reviewPanel()),
      copy: "Try choosing another cut here. BeatLedger updates the source excerpt, exact timecodes, scores and editor controls for that cut.",
    },
    {
      title: "Read the full cut", tab: "Review cuts", open: "Full cut excerpt", above: true,
      target: () => disclosure("Full cut excerpt"),
      copy: "Read the dialogue and actions in order. Compare the opening and ending with the nearby Source evidence to see why the cut was picked.",
    },
    {
      title: "Preview a trim", tab: "Review cuts", open: "Adjust cut", above: true,
      target: () => visible('[class*="st-key-trim-range-"]', reviewPanel()),
      copy: "Try dragging either handle. Read the updated excerpt on the left, then click Verify and apply trim when you’re ready to save the new range.",
    },
    {
      title: "Save an editor decision", tab: "Review cuts",
      target: () => visible('[data-testid="stForm"]', reviewPanel()),
      copy: "Choose Pending, Approved or Rejected and add a note for the next editor. Save decision applies to the current saved cut range.",
    },
    {
      title: "Download what you need", tab: "Review cuts", open: "Download run and reviews",
      target: () => disclosure("Download run and reviews")?.parentElement,
      copy: "Use Ranked CSV to scan the list and Ranked JSON to inspect its evidence. Review decisions JSON becomes available after you save a decision or trim.",
    },
    {
      title: "Check the run metrics", tab: "Metrics",
      target: () => tab("Metrics"),
      copy: "Check how many episodes got a cut, how much work the model did, and whether model judging changed scores or ranks.",
    },
    {
      title: "Trace a warning in Run audit", tab: "Run audit",
      target: () => tab("Run audit"),
      copy: "See which files loaded, where episodes were joined, and why anything was flagged. Download the run ledger or review flags to share the details.",
    },
  ];

  const overlay = document.createElement("div");
  overlay.id = "bl-tour";
  overlay.hidden = true;
  overlay.innerHTML = [
    '<div class="bl-tour-spotlight" aria-hidden="true" hidden></div>',
    '<section class="bl-tour-card" role="dialog" aria-labelledby="bl-tour-title" aria-describedby="bl-tour-copy">',
    '<div class="bl-tour-heading"><span class="bl-tour-step"></span>',
    '<button type="button" data-tour-action="close" aria-label="Close tour">×</button></div>',
    '<h2 id="bl-tour-title" tabindex="-1"></h2><p id="bl-tour-copy"></p>',
    '<p class="bl-tour-missing" hidden>Run the bundled sample to try this control.</p>',
    '<div class="bl-tour-actions"><button type="button" data-tour-action="skip">End tour</button>',
    '<button type="button" data-tour-action="back">Back</button>',
    '<button type="button" data-tour-action="next">Next</button></div></section>',
  ].join("");
  document.body.append(overlay);

  const card = overlay.querySelector(".bl-tour-card");
  const spotlight = overlay.querySelector(".bl-tour-spotlight");
  const heading = card.querySelector("h2");
  const next = card.querySelector('[data-tour-action="next"]');
  const back = card.querySelector('[data-tour-action="back"]');
  const missing = card.querySelector(".bl-tour-missing");
  const observer = new MutationObserver(() => schedulePosition());
  let index = 0;
  let frame = 0;
  let visit = 0;
  let startingTab;

  function position() {
    const step = steps[index];
    const box = step.target()?.getBoundingClientRect();
    const width = document.documentElement.clientWidth;
    const height = window.innerHeight;
    const margin = 12;
    const clamp = (n, high) => Math.max(margin, Math.min(n, high - margin));
    missing.hidden = Boolean(box);
    const w = card.offsetWidth;
    const h = card.offsetHeight;
    let x = width - w - margin;
    let y = height - h - margin;

    const inView = box && box.right > 0 && box.bottom > 0 && box.left < width && box.top < height;
    spotlight.hidden = !inView;
    if (inView) {
      const left = Math.max(2, box.left - 5);
      const top = Math.max(2, box.top - 5);
      const right = Math.min(width - 2, box.right + 5);
      const bottom = Math.min(height - 2, box.bottom + 5);
      Object.assign(spotlight.style, {
        left: left + "px", top: top + "px", width: (right - left) + "px", height: (bottom - top) + "px",
      });
      const gap = 14;
      const sidebarEdge = step.sidebar && visible('[data-testid="stSidebar"]')?.getBoundingClientRect().right;
      if (sidebarEdge && width - sidebarEdge >= w + gap + margin) {
        x = sidebarEdge + gap;
        y = top;
      } else if (step.above && top >= h + gap + margin) {
        x = right - w;
        y = top - h - gap;
      } else if (width - right >= w + gap + margin) {
        x = right + gap;
        y = top;
      } else if (left >= w + gap + margin) {
        x = left - w - gap;
        y = top;
      } else {
        x = left + (right - left - w) / 2;
        y = bottom + gap + h + margin <= height ? bottom + gap : top - h - gap;
      }
    }
    card.style.left = clamp(x, width - w) + "px";
    card.style.top = clamp(y, height - h) + "px";
  }

  function schedulePosition() {
    if (frame || overlay.hidden) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      if (!overlay.hidden) position();
    });
  }

  function showStep(number) {
    index = number;
    const step = steps[number];
    const token = ++visit;
    heading.textContent = step.title;
    card.querySelector("#bl-tour-copy").textContent = step.copy;
    card.querySelector(".bl-tour-step").textContent = (number + 1) + " of " + steps.length;
    back.disabled = number === 0;
    next.textContent = number === steps.length - 1 ? "Finish" : "Next";
    heading.focus({ preventScroll: true });
    if (step.sidebar) {
      visible('button[aria-label="Expand sidebar"]')?.click();
    } else if (window.matchMedia("(max-width: 768px)").matches) {
      visible('button[aria-label="Collapse sidebar"]')?.click();
    }
    const selectedTab = step.tab && tab(step.tab);
    if (selectedTab && selectedTab.getAttribute("aria-selected") !== "true") selectedTab.click();

    let opened = false;
    function find(attempt) {
      if (token !== visit || overlay.hidden) return;
      const summary = step.open && disclosure(step.open);
      if (summary && !opened) {
        if (!summary.parentElement.open) summary.click();
        opened = true;
      }
      const target = step.target();
      if (!target && attempt < 12) {
        setTimeout(() => find(attempt + 1), 50);
        return;
      }
      target?.scrollIntoView({ block: "center", inline: "nearest", behavior: "instant" });
      schedulePosition();
      setTimeout(schedulePosition, 300);
    }
    requestAnimationFrame(() => find(0));
    schedulePosition();
  }

  function close() {
    overlay.hidden = true;
    visit++;
    observer.disconnect();
    document.removeEventListener("scroll", schedulePosition, true);
    window.removeEventListener("resize", schedulePosition);
    const previous = startingTab && tab(startingTab);
    if (previous && previous.getAttribute("aria-selected") !== "true") previous.click();
    const returnFocus = visible('button[aria-label="Expand sidebar"]') || visible("#bl-tour-launch");
    returnFocus?.focus({ preventScroll: true });
  }

  document.addEventListener("click", event => {
    if (!event.target.closest("#bl-tour-launch") || !overlay.hidden) return;
    startingTab = document.querySelector('[role="tab"][aria-selected="true"]')?.textContent.trim();
    overlay.hidden = false;
    document.addEventListener("scroll", schedulePosition, true);
    window.addEventListener("resize", schedulePosition);
    const app = document.querySelector('[data-testid="stApp"]');
    if (app) observer.observe(app, { childList: true, subtree: true });
    showStep(0);
  });
  card.addEventListener("click", event => {
    const action = event.target.closest("[data-tour-action]")?.dataset.tourAction;
    if (action === "next" && index < steps.length - 1) showStep(index + 1);
    else if (action === "back" && index > 0) showStep(index - 1);
    else if (action) close();
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape" && !overlay.hidden) {
      event.preventDefault();
      close();
    }
  });
}
