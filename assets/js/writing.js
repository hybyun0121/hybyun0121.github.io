(function () {
  'use strict';

  var index = document.querySelector('[data-writing-index]');
  if (index) {
    var filters = index.querySelector('.writing-filters');
    var buttons = Array.from(index.querySelectorAll('[data-writing-filter]'));
    var cards = Array.from(index.querySelectorAll('[data-writing-type]'));
    var empty = index.querySelector('[data-writing-empty]');
    filters.hidden = false;
    buttons.forEach(function (button) {
      button.addEventListener('click', function () {
        var filter = button.dataset.writingFilter;
        var count = 0;
        buttons.forEach(function (item) { item.setAttribute('aria-pressed', String(item === button)); });
        cards.forEach(function (card) {
          card.hidden = filter !== 'all' && card.dataset.writingType !== filter;
          if (!card.hidden) count += 1;
        });
        empty.hidden = count > 0;
        empty.textContent = filter === 'all' ? '아직 공개된 글이 없습니다.' : '이 분류에는 아직 공개된 글이 없습니다.';
      });
    });
  }

  var body = document.querySelector('[data-writing-body]');
  var toc = document.querySelector('[data-writing-toc]');
  if (!body || !toc) return;

  // Long tables scroll within the reading column, including on narrow screens.
  body.querySelectorAll('table').forEach(function (table) {
    var wrapper = document.createElement('div');
    wrapper.className = 'writing-table';
    wrapper.tabIndex = 0;
    wrapper.setAttribute('role', 'region');
    wrapper.setAttribute('aria-label', 'Table (scroll horizontally if needed)');
    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });

  function revealHashTarget() {
    var id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); }
    catch (error) { return; }
    var target = id && document.getElementById(id);
    if (!target || !body.contains(target)) return;
    var detail = target.tagName === 'DETAILS' ? target : target.closest('details');
    while (detail) {
      detail.open = true;
      detail = detail.parentElement.closest('details');
    }
    target.scrollIntoView();
  }
  window.addEventListener('hashchange', revealHashTarget);
  if (window.location.hash) revealHashTarget();

  var headings = Array.from(body.querySelectorAll('h2, h3')).filter(function (heading) {
    return !heading.closest('details');
  });
  if (!headings.length) return;
  var list = toc.querySelector('ol');
  var links = [];
  headings.forEach(function (heading, i) {
    if (!heading.id) {
      var candidate = 'section-' + (i + 1);
      while (document.getElementById(candidate)) candidate += '-section';
      heading.id = candidate;
    }
    var item = document.createElement('li');
    var link = document.createElement('a');
    if (heading.tagName === 'H3') item.className = 'writing-toc__sub';
    link.href = '#' + encodeURIComponent(heading.id);
    link.textContent = heading.textContent;
    item.appendChild(link);
    list.appendChild(item);
    links.push(link);
  });
  toc.hidden = false;
  toc.querySelector('details').open = window.matchMedia('(min-width: 1100px)').matches;
  if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
    window.MathJax.startup.promise.then(function () {
      return window.MathJax.typesetPromise([toc]);
    }).catch(function () { /* The text contents still provide usable navigation. */ });
  }

  if ('IntersectionObserver' in window) {
    var visible = new Set();
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
      });
      var active = headings.find(function (heading) { return visible.has(heading); });
      if (!active) return;
      links.forEach(function (link, i) {
        if (headings[i] === active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    }, {rootMargin: '-12% 0px -55% 0px'});
    headings.forEach(function (heading) { observer.observe(heading); });
  }
}());
