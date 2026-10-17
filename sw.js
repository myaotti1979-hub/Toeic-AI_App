const CACHE_NAME = 'toeic-trainer-v2026.10.17a';
const FONT_CACHE = 'toeic-trainer-fonts-v1';   // kept across app updates
const CORE_ASSETS = ['./', './index.html', './manifest.json', './icon-192.png', './icon-512.png'];

// Install: pre-cache core assets. A missing file must not block the update,
// so each asset is cached on its own and failures are ignored.
self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => Promise.all(CORE_ASSETS.map(u =>
        cache.add(u).catch(err => console.warn('[SW] precache skip', u, err))
      )))
      .then(() => self.skipWaiting())
  );
});

// Activate: delete old app caches (keep the font cache), take control immediately
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME && k !== FONT_CACHE).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;          // never cache POST etc.
  const url = new URL(e.request.url);

  // Google Fonts (Noto Sans JP) — cache-first so Japanese text keeps the
  // correct font offline instead of falling back to a system CJK font
  if (url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') {
    e.respondWith(
      caches.open(FONT_CACHE).then(cache =>
        cache.match(e.request).then(hit => hit || fetch(e.request).then(resp => {
          if (resp && (resp.ok || resp.type === 'opaque')) cache.put(e.request, resp.clone());
          return resp;
        }))
      )
    );
    return;
  }

  // Other cross-origin calls (Gemini, Azure, …) — always network, never cache
  if (url.hostname !== location.hostname) return;

  // Page navigation / index.html — NETWORK FIRST so an update is picked up on
  // the next launch; offline falls back to the cached copy of the app shell
  if (e.request.mode === 'navigate' || url.pathname.endsWith('index.html') || url.pathname.endsWith('/')) {
    e.respondWith(
      fetch(e.request)
        .then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(e.request, clone));
          }
          return response;
        })
        .catch(() =>
          caches.match(e.request, {ignoreSearch: true})
            .then(hit => hit || caches.match('./index.html'))
        )
    );
    return;
  }

  // Other same-origin assets (icons, manifest) — cache first, refresh in background
  e.respondWith(
    caches.match(e.request).then(cached => {
      const fetchPromise = fetch(e.request).then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(e.request, clone));
        }
        return response;
      }).catch(() => cached);
      return cached || fetchPromise;
    })
  );
});
