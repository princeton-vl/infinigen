function attachCopy(btn, getText, resting) {
  btn.addEventListener("click", () => {
    navigator.clipboard
      .writeText(getText())
      .then(() => {
        btn.classList.add("copied");
        btn.textContent = "✓";
        setTimeout(() => {
          btn.classList.remove("copied");
          btn.textContent = resting;
        }, 1200);
      })
      .catch(() => {
        btn.textContent = "✗";
        setTimeout(() => {
          btn.textContent = resting;
        }, 1200);
      });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".example-render__copy").forEach((btn) => {
    const code = btn.parentElement.querySelector(".example-render__code");
    const getText = () => btn.dataset.command || (code ? code.textContent : "");
    attachCopy(btn, getText, btn.textContent);
  });

  document.querySelectorAll(".example-card pre").forEach((pre) => {
    if (pre.querySelector(".example-card__copy")) return;
    const code = pre.querySelector("code") || pre;
    const text = code.textContent;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "example-card__copy";
    btn.textContent = "⧉";
    btn.setAttribute("aria-label", "Copy command");
    pre.appendChild(btn);
    attachCopy(btn, () => text, "⧉");
  });
});
