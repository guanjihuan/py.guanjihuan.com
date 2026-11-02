/* ==========================================================================
   GUAN 官网 —— 交互脚本
   站点是纯静态的，所以所有交互都放在这里，按功能分成若干独立的小模块。
   每个模块都先判断页面上有没有对应的元素，没有就直接退出，
   这样同一份脚本可以用在所有页面上。
   ========================================================================== */

(function () {
  "use strict";

  var root = document.documentElement;

  /* 页面把自己的语言文案传进来（见 i18n.py），
     这里做一次兜底，脚本单独拿出来调试时也不会报错。 */
  var I18N = window.SITE_I18N || {};
  function text(key, fallback) {
    return I18N[key] || fallback || key;
  }

  /* ---------- 通用小工具 ---------- */

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.appendChild(area);
      area.select();
      try {
        document.execCommand("copy");
        resolve();
      } catch (err) {
        reject(err);
      }
      document.body.removeChild(area);
    });
  }

  /* 给按钮加一个「已复制 → 恢复」的反馈 */
  function flash(button, text, done) {
    var original = button.getAttribute("data-label") || button.textContent;
    button.setAttribute("data-label", original);
    button.textContent = text;
    clearTimeout(button._flashTimer);
    button._flashTimer = setTimeout(function () {
      button.textContent = original;
      if (done) done();
    }, 1500);
  }

  /* ==========================================================================
     主题切换：默认暗色，用户点一次就固定下来
     ========================================================================== */

  (function theme() {
    var button = document.getElementById("theme-toggle");
    if (!button) return;

    function current() {
      var explicit = root.getAttribute("data-theme");
      if (explicit === "light" || explicit === "dark") return explicit;
      return "dark";
    }

    function apply(mode) {
      if (mode === "light" || mode === "dark") {
        root.setAttribute("data-theme", mode);
      } else {
        root.removeAttribute("data-theme");
      }
    }

    button.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      apply(next);
      try {
        localStorage.setItem("guan-theme", next);
      } catch (e) {}
      button.setAttribute(
        "title",
        next === "dark"
          ? text("theme_to_light", "切换到浅色主题")
          : text("theme_to_dark", "切换到深色主题")
      );
      button.setAttribute("aria-label", button.getAttribute("title"));
    });
  })();

  /* ==========================================================================
     窄屏菜单：展开 / 收起导航
     ========================================================================== */

  (function navToggle() {
    var toggle = document.getElementById("nav-toggle");
    var nav = document.getElementById("site-nav");
    if (!toggle || !nav) return;

    function close() {
      nav.classList.remove("is-open");
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-label", text("menu_open", "打开菜单"));
    }

    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("is-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      toggle.setAttribute(
        "aria-label",
        open ? text("menu_close", "关闭菜单") : text("menu_open", "打开菜单")
      );
    });

    nav.addEventListener("click", function (e) {
      if (e.target.tagName === "A") close();
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });

    window.addEventListener("resize", function () {
      if (window.innerWidth > 900) close();
    });
  })();

  /* ==========================================================================
     复制：安装命令、代码块、函数链接
     ========================================================================== */

  (function copyButtons() {
    document.querySelectorAll(".copy[data-copy]").forEach(function (button) {
      button.addEventListener("click", function () {
        var value = button.getAttribute("data-copy");
        copyText(value).then(function () {
          flash(button, text("copied", "已复制"));
        }, function () {});
      });
    });

    // 给每个代码块加一个右上角的复制按钮
    document.querySelectorAll("pre.code").forEach(function (pre) {
      var code = pre.querySelector("code");
      if (!code) return;
      var button = document.createElement("button");
      button.type = "button";
      button.className = "code-copy";
      button.textContent = text("copy", "复制");
      button.title = text("copy_code", "复制代码");
      button.setAttribute("aria-label", text("copy_code", "复制代码"));
      button.addEventListener("click", function () {
        copyText(code.innerText).then(function () {
          flash(button, text("copied", "已复制"));
        }, function () {});
      });
      pre.appendChild(button);
    });
  })();

  /* ==========================================================================
     下载：教程页的「下载脚本」
     直接用 <a download> 的话，本地以 file:// 方式预览时浏览器会忽略
     download 属性、把源码显示出来。这里改成拿页面里的源码生成文件再下载，
     这样无论站点怎么打开，点击都会启动下载程序。
     ========================================================================== */

  (function downloadLinks() {
    document.querySelectorAll("a[data-download]").forEach(function (link) {
      link.addEventListener("click", function (event) {
        var card = link.closest(".tut-card");
        var code = card && card.querySelector("pre.code code");
        if (!code) return; // 页面里没有源码，就用浏览器自己的下载行为

        event.preventDefault();
        var blob = new Blob([code.textContent], {
          type: "text/x-python;charset=utf-8"
        });
        var url = URL.createObjectURL(blob);
        var anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = link.getAttribute("data-download");
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
      });
    });
  })();

  /* ==========================================================================
     回到顶部
     ========================================================================== */

  (function toTop() {
    var button = document.getElementById("to-top");
    if (!button) return;

    var pending = false;

    function update() {
      pending = false;
      button.classList.toggle("is-visible", window.scrollY > 420);
    }

    function onScroll() {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(update);
    }

    button.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  })();

  /* ==========================================================================
     滚动跟随高亮：API 页的模块列表、关于页 / 教程页的目录
     --------------------------------------------------------------------------
     侧栏里最后一个条目往往太短，页面滚不到它的标题上方那条基线，
     只按滚动位置判断的话，高亮会留在上一项上，点它也没反应。
     所以点一下的时候先把高亮锁在被点的条目上，等用户自己滚动时
     再交还给滚动位置决定；页面滚到底部时，最后一个条目也会被点亮。
     ========================================================================== */

  function setupScrollSpy(options) {
    var links = options.links;
    var targets = options.targets;
    var offset = options.offset || 120;
    var container = options.container || null;

    if (!links.length || !targets.length) return;

    var locked = null; // 点击后锁定的目标
    var settledY = null; // 跳转停下来时的滚动位置
    var settleTimer = null;
    var pending = false;

    function hrefId(link) {
      var href = link.getAttribute("href") || "";
      if (href.charAt(0) !== "#") return "";
      return decodeURIComponent(href.slice(1));
    }

    function isVisible(el) {
      return el.getClientRects().length > 0;
    }

    /* 当前所在的目标：最后一个越过基线的；页面到底了就是最后一个 */
    function pick() {
      var doc = document.documentElement;
      var shown = targets.filter(isVisible);
      if (!shown.length) return null;
      if (window.innerHeight + window.pageYOffset >= doc.scrollHeight - 4) {
        return shown[shown.length - 1];
      }
      var current = null;
      shown.forEach(function (target) {
        if (target.getBoundingClientRect().top <= offset) current = target;
      });
      return current || shown[0];
    }

    /* 让被点亮的条目留在侧栏自身的可视范围内 */
    function keepVisible(link) {
      if (!container) return;
      var box = container.getBoundingClientRect();
      var item = link.getBoundingClientRect();
      if (item.top < box.top) container.scrollTop -= box.top - item.top + 6;
      else if (item.bottom > box.bottom) container.scrollTop += item.bottom - box.bottom + 6;
    }

    function apply(target) {
      links.forEach(function (link) {
        var active = !!target && hrefId(link) === target.id;
        link.classList.toggle("is-active", active);
        if (active) keepVisible(link);
      });
    }

    /* 滚动停下来之后记下位置，用来分辨「程序滚动」和「用户滚动」 */
    function markSettled() {
      clearTimeout(settleTimer);
      settleTimer = setTimeout(function () {
        settledY = window.scrollY;
        apply(locked || pick());
      }, 200);
    }

    function update() {
      pending = false;
      // 用户自己滚动了 → 解除点击时的锁定
      if (locked && settledY !== null && Math.abs(window.scrollY - settledY) > 4) {
        locked = null;
        settledY = null;
      }
      apply(locked || pick());
      markSettled();
    }

    function onScroll() {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(update);
    }

    links.forEach(function (link) {
      link.addEventListener("click", function () {
        var id = hrefId(link);
        if (!id) return;
        var target = document.getElementById(id);
        if (!target) return;
        locked = target;
        settledY = null;
        apply(target);
        markSettled();
      });
    });

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    apply(pick());
    markSettled();
  }

  /* ==========================================================================
     API 页：模块速查、模块高亮、函数筛选
     ========================================================================== */

  (function apiPage() {
    var layout = document.querySelector(".api-layout");
    if (!layout) return;

    var modules = Array.prototype.slice.call(document.querySelectorAll(".module"));

    /* 每个模块下方加一排函数快捷入口 */
    modules.forEach(function (module) {
      var funcs = Array.prototype.slice.call(module.querySelectorAll(".func"));
      if (funcs.length < 2) return;
      var list = document.createElement("ul");
      list.className = "module-index";
      funcs.forEach(function (func) {
        var id = func.id;
        if (!id) return;
        var name = id.split(".").pop();
        var li = document.createElement("li");
        var link = document.createElement("a");
        link.href = "#" + id;
        link.textContent = name;
        link.setAttribute("data-target", id);
        li.appendChild(link);
        list.appendChild(li);
      });
      var meta = module.querySelector(".module-meta");
      if (meta) meta.insertAdjacentElement("afterend", list);
      else module.insertBefore(list, module.firstChild);
    });

    /* 函数卡片右上角的「复制链接」按钮 */
    document.querySelectorAll(".func").forEach(function (func) {
      if (!func.id) return;
      var button = document.createElement("button");
      button.type = "button";
      button.className = "anchor-btn";
      button.textContent = "#";
      button.title = text("copy_anchor", "复制本函数的链接");
      button.setAttribute("aria-label", text("copy_anchor", "复制本函数的链接"));
      button.addEventListener("click", function () {
        var url = location.href.split("#")[0] + "#" + func.id;
        copyText(url).then(function () {
          flash(button, text("copied", "已复制"));
        }, function () {});
      });
      var heading = func.querySelector("h3");
      if (heading) heading.appendChild(button);
    });

    /* 顶部筛选框：按函数名 / 说明过滤 */
    var total = document.querySelectorAll(".func").length;
    var filter = document.createElement("div");
    filter.className = "api-filter";
    filter.innerHTML =
      '<input type="search" placeholder="' +
      text("filter_placeholder", "在本页筛选函数…") +
      '" aria-label="' + text("filter_label", "筛选函数") +
      '" autocomplete="off">' +
      '<span class="filter-count"></span>' +
      '<button class="filter-clear" type="button" hidden>' +
      text("filter_clear", "清除") + "</button>";
    // 放在正文列的上方，跟左侧模块列表对齐
    var body = document.querySelector(".api-body");
    if (body) body.insertBefore(filter, body.firstChild);
    else layout.parentNode.insertBefore(filter, layout);

    var field = filter.querySelector("input");
    var count = filter.querySelector(".filter-count");
    var clear = filter.querySelector(".filter-clear");

    function applyFilter() {
      var q = field.value.trim().toLowerCase();
      clear.hidden = !q;
      var shown = 0;

      document.querySelectorAll(".func").forEach(function (func) {
        var content = (func.textContent || "").toLowerCase();
        var hit = !q || content.indexOf(q) !== -1;
        func.classList.toggle("is-hidden", !hit);
        if (hit) shown += 1;
      });

      // 一个函数都不剩的模块整块隐藏
      modules.forEach(function (module) {
        var visible = module.querySelectorAll(".func:not(.is-hidden)").length;
        module.style.display = visible ? "" : "none";
        var index = module.querySelector(".module-index");
        if (index) {
          index.querySelectorAll("a").forEach(function (link) {
            var target = document.getElementById(link.getAttribute("data-target"));
            link.style.display = target && !target.classList.contains("is-hidden") ? "" : "none";
          });
        }
      });

      count.textContent = q
        ? text("filter_count", "显示 {shown} / {total} 个函数")
            .replace("{shown}", shown)
            .replace("{total}", total)
        : "";
    }

    field.addEventListener("input", applyFilter);
    clear.addEventListener("click", function () {
      field.value = "";
      applyFilter();
      field.focus();
    });

    /* 左侧模块列表：滚动时高亮当前所在的模块 */
    var nav = document.querySelector(".api-nav");
    var navLinks = Array.prototype.slice.call(
      document.querySelectorAll(".api-nav li a")
    );
    if (!navLinks.length) return;

    setupScrollSpy({
      links: navLinks,
      targets: modules,
      offset: 120,
      container: nav,
    });
  })();

  /* ==========================================================================
     关于页 / 教程页的目录：跟随滚动高亮当前章节
     ========================================================================== */

  (function tocScrollSpy() {
    var toc = document.querySelector(".toc");
    if (!toc) return;

    var links = Array.prototype.slice.call(toc.querySelectorAll("a[href^='#']"));
    if (!links.length) return;

    var targets = links
      .map(function (link) {
        return document.getElementById(decodeURIComponent(link.getAttribute("href").slice(1)));
      })
      .filter(Boolean);
    if (!targets.length) return;

    setupScrollSpy({
      links: links,
      targets: targets,
      offset: 140,
      container: toc,
    });
  })();
})();
