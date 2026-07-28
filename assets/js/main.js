(function () {
  var reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var root = document.documentElement;

  // Theme toggle. Light is the default. Circular reveal via View Transitions API.
  var toggles = document.querySelectorAll('.theme-toggle');
  if (toggles.length) {
    var isDark = function () { return root.classList.contains('theme-dark'); };
    var sync = function () {
      var dark = isDark();
      toggles.forEach(function (tgl) {
        tgl.querySelector('.ico-sun').style.display = dark ? 'inline-flex' : 'none';
        tgl.querySelector('.ico-moon').style.display = dark ? 'none' : 'inline-flex';
      });
    };
    sync();
    toggles.forEach(function (tgl) {
      tgl.addEventListener('click', function () {
        var next = isDark() ? 'light' : 'dark';
        var apply = function () {
          root.classList.toggle('theme-dark', next === 'dark');
          try { localStorage.setItem('theme', next); } catch (e) {}
          sync();
        };
        if (!document.startViewTransition || reduced) { apply(); return; }
        var rect = tgl.getBoundingClientRect();
        var x = rect.left + rect.width / 2;
        var y = rect.top + rect.height / 2;
        var r = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
        document.startViewTransition(apply).ready.then(function () {
          root.animate(
            { clipPath: ['circle(0px at ' + x + 'px ' + y + 'px)', 'circle(' + r + 'px at ' + x + 'px ' + y + 'px)'] },
            { duration: 550, easing: 'ease-in-out', pseudoElement: '::view-transition-new(root)' }
          );
        });
      });
    });
  }

  // Scroll reveals
  var reveals = document.querySelectorAll('.reveal');
  if (reveals.length && 'IntersectionObserver' in window && !reduced) {
    var ro = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); ro.unobserve(en.target); }
      });
    }, { threshold: 0.01 });
    reveals.forEach(function (el) {
      if (el.getBoundingClientRect().top < window.innerHeight * 1.1) {
        el.classList.add('in');
      } else {
        ro.observe(el);
      }
    });
  } else {
    reveals.forEach(function (el) { el.classList.add('in'); });
  }

  // Card spotlight (cursor-follow glow)
  if (!reduced) {
    document.querySelectorAll('.card').forEach(function (card) {
      card.addEventListener('pointermove', function (e) {
        var rect = card.getBoundingClientRect();
        card.style.setProperty('--mx', (e.clientX - rect.left) + 'px');
        card.style.setProperty('--my', (e.clientY - rect.top) + 'px');
      });
    });
  }

  // Citation counts — Google Scholar data only (daily crawl deployed to the
  // google-scholar-stats branch, served via jsDelivr; bundled snapshot as backup).
  (function () {
    var els = document.querySelectorAll('.pub-cite');
    if (!els.length || !window.fetch) return;
    var KEY = 'gs-cites-v3';
    var norm = function (s) {
      return (s || '').toLowerCase().replace(/\$[^$]*\$/g, '').replace(/[^a-z0-9]/g, '');
    };
    var totalEl = document.querySelector('.cite-total');
    var setTotal = function (n) {
      if (totalEl && n > 0) {
        totalEl.querySelector('b').textContent = Number(n).toLocaleString();
        totalEl.hidden = false;
      }
    };
    var setEl = function (el, n) {
      el.querySelector('b').textContent = Number(n).toLocaleString();
      el.hidden = false;
    };
    var applyMap = function (data) {
      setTotal(data.total);
      els.forEach(function (el) {
        var k = norm(el.getAttribute('data-title'));
        if (k && k in data.map) setEl(el, data.map[k]);
      });
    };
    var save = function (data) {
      data.t = Date.now();
      try { localStorage.setItem(KEY, JSON.stringify(data)); } catch (e) {}
    };
    try {
      var c = JSON.parse(localStorage.getItem(KEY) || 'null');
      if (c && Date.now() - c.t < 43200000) { applyMap(c); return; }
    } catch (e) {}

    var applyGs = function (gs) {
      var pubs = gs.publications || {};
      var gsList = Object.keys(pubs).map(function (k) {
        return { t: norm(pubs[k].bib && pubs[k].bib.title), n: pubs[k].num_citations || 0 };
      });
      var map = {};
      els.forEach(function (el) {
        var title = el.getAttribute('data-title') || '';
        var key = norm(title);
        if (!key || key.length < 8) return;
        var nick = norm(title.split(':')[0]);
        for (var i = 0; i < gsList.length; i++) {
          var g = gsList[i];
          if (!g.t) continue;
          if (g.t === key || g.t.indexOf(key) !== -1 || key.indexOf(g.t) !== -1 ||
              (nick.length >= 5 && g.t.indexOf(nick) === 0)) {
            map[key] = g.n;
            break;
          }
        }
      });
      var data = { total: gs.citedby || 0, map: map };
      save(data);
      applyMap(data);
    };

    // Freshest first: daily-crawled branch via jsDelivr, then the snapshot bundled with the site.
    fetch('https://cdn.jsdelivr.net/gh/HaoyiZhu/homepage@google-scholar-stats/gs_data.json')
      .then(function (r) { if (!r.ok) throw new Error('no gs data'); return r.json(); })
      .then(applyGs)
      .catch(function () {
        fetch('/data/gs_data.json')
          .then(function (r) { if (!r.ok) throw new Error('no snapshot'); return r.json(); })
          .then(applyGs)
          .catch(function () {});
      });
  })();

  // GitHub stars, cached 12h per repo
  (function () {
    var els = document.querySelectorAll('.gh-stars[data-repo]');
    if (!els.length || !window.fetch) return;
    var fmt = function (n) { return n >= 1000 ? (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k' : String(n); };
    var byRepo = {};
    els.forEach(function (el) {
      var repo = el.getAttribute('data-repo');
      (byRepo[repo] = byRepo[repo] || []).push(el);
    });
    Object.keys(byRepo).forEach(function (repo) {
      var KEY = 'gh-stars:' + repo;
      var show = function (n) {
        byRepo[repo].forEach(function (el) { el.querySelector('b').textContent = fmt(n); el.hidden = false; });
      };
      try {
        var c = JSON.parse(localStorage.getItem(KEY) || 'null');
        if (c && Date.now() - c.t < 43200000) { show(c.n); return; }
      } catch (e) {}
      fetch('https://api.github.com/repos/' + repo).then(function (r) { return r.json(); }).then(function (d) {
        if (d && typeof d.stargazers_count === 'number') {
          try { localStorage.setItem(KEY, JSON.stringify({ t: Date.now(), n: d.stargazers_count })); } catch (e) {}
          show(d.stargazers_count);
        }
      }).catch(function () {});
    });
  })();

  // Blog: reading progress bar
  var prog = document.querySelector('.blog-progress span');
  if (prog) {
    var updateProg = function () {
      var h = document.documentElement.scrollHeight - window.innerHeight;
      prog.style.width = (h > 0 ? (window.scrollY / h) * 100 : 0) + '%';
    };
    window.addEventListener('scroll', updateProg, { passive: true });
    updateProg();
  }

  // Blog: TOC scrollspy (position-based)
  var tocLinks = document.querySelectorAll('.blog-toc a[href^="#"]');
  if (tocLinks.length) {
    var tocHeads = [];
    tocLinks.forEach(function (a) {
      var el = document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
      if (el) tocHeads.push({ el: el, link: a });
    });
    tocHeads.sort(function (a, b) { return a.el.offsetTop - b.el.offsetTop; });
    var tocCurrent = null;
    var spy = function () {
      var y = window.scrollY + window.innerHeight * 0.28;
      var cur = null;
      for (var i = 0; i < tocHeads.length; i++) {
        if (tocHeads[i].el.offsetTop <= y) cur = tocHeads[i].link;
        else break;
      }
      if (cur !== tocCurrent) {
        if (tocCurrent) tocCurrent.classList.remove('active');
        if (cur) cur.classList.add('active');
        tocCurrent = cur;
      }
    };
    window.addEventListener('scroll', spy, { passive: true });
    spy();
  }

  // Nav scrollspy
  var navLinks = document.querySelectorAll('.nav-links a[href^="/#"]');
  if (navLinks.length && 'IntersectionObserver' in window) {
    var byId = {};
    navLinks.forEach(function (a) { byId[a.getAttribute('href').slice(2)] = a; });
    var current = null;
    var so = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && byId[en.target.id]) {
          if (current) current.classList.remove('active');
          current = byId[en.target.id];
          current.classList.add('active');
        }
      });
    }, { rootMargin: '-30% 0px -55% 0px' });
    Object.keys(byId).forEach(function (id) {
      var sec = document.getElementById(id);
      if (sec) so.observe(sec);
    });
  }

  // News toggle
  var toggle = document.querySelector('.news-toggle');
  if (toggle) {
    var list = document.querySelector('.news-list');
    var label = toggle.textContent;
    toggle.addEventListener('click', function () {
      var open = list.classList.toggle('open');
      toggle.textContent = open ? 'Show fewer' : label;
    });
  }

  // Paintings: vertical wheel -> horizontal scroll, plus pointer drag
  var scroll = document.querySelector('.handscroll');
  if (scroll) {
    scroll.addEventListener('wheel', function (e) {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        var atStart = scroll.scrollLeft <= 0 && e.deltaY < 0;
        var atEnd = scroll.scrollLeft + scroll.clientWidth >= scroll.scrollWidth - 1 && e.deltaY > 0;
        if (!atStart && !atEnd) {
          e.preventDefault();
          scroll.scrollLeft += e.deltaY;
        }
      }
    }, { passive: false });

    var dragging = false, startX = 0, startLeft = 0, moved = false;
    scroll.addEventListener('pointerdown', function (e) {
      if (e.pointerType !== 'mouse') return;
      dragging = true; moved = false;
      startX = e.clientX; startLeft = scroll.scrollLeft;
      scroll.classList.add('dragging');
    });
    window.addEventListener('pointermove', function (e) {
      if (!dragging) return;
      var dx = e.clientX - startX;
      if (Math.abs(dx) > 4) moved = true;
      scroll.scrollLeft = startLeft - dx;
    });
    window.addEventListener('pointerup', function () {
      dragging = false;
      scroll.classList.remove('dragging');
    });
    scroll.addEventListener('click', function (e) {
      if (moved) { e.preventDefault(); e.stopPropagation(); moved = false; }
    }, true);
  }

  // Paintings lightbox
  var gLinks = document.querySelectorAll('.handscroll a');
  if (gLinks.length) {
    var lb = document.createElement('div');
    lb.className = 'lightbox';
    var lbImg = document.createElement('img');
    lbImg.alt = 'Traditional Chinese painting';
    var lbClose = document.createElement('span');
    lbClose.className = 'lightbox-close';
    lbClose.textContent = '×';
    lb.appendChild(lbImg);
    lb.appendChild(lbClose);
    document.body.appendChild(lb);
    var close = function () {
      lb.classList.remove('open');
      lbImg.src = '';
      document.body.style.overflow = '';
    };
    gLinks.forEach(function (a) {
      a.addEventListener('click', function (e) {
        e.preventDefault();
        lbImg.src = a.href;
        lb.classList.add('open');
        document.body.style.overflow = 'hidden';
      });
    });
    lb.addEventListener('click', close);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
  }
})();
