```js
// ==UserScript==
// @name         ChatGPT Compact Markdown Copier v4.5.0
// @namespace    https://chatgpt.com/
// @version      5.3.0
// @description  为每条 ChatGPT 回复添加 Markdown 复制按钮；自动区分行内公式 $ 与块级公式 $$，修复公式中的 < 和 >，删除图片并整理 Markdown 格式。
// @match        https://chatgpt.com/*
// @match        https://chat.openai.com/*
// @grant        GM_setClipboard
// @run-at       document-idle
// ==/UserScript==

(() => {
  'use strict';

  const BUTTON_CLASS = 'tm-copy-compact-markdown';
  const WRAPPER_CLASS = 'tm-copy-compact-markdown-wrapper';
  const TOAST_ID = 'tm-compact-markdown-toast';
  const GLOBAL_BUTTON_ID = 'tm-copy-last-compact-markdown';

  function showToast(text) {
    document.getElementById(TOAST_ID)?.remove();

    const toast = document.createElement('div');
    toast.id = TOAST_ID;
    toast.textContent = text;

    Object.assign(toast.style, {
      position: 'fixed',
      right: '20px',
      bottom: '20px',
      zIndex: '2147483647',
      padding: '10px 14px',
      borderRadius: '8px',
      background: '#202123',
      color: '#ffffff',
      fontSize: '13px',
      lineHeight: '1.4',
      boxShadow: '0 4px 18px rgba(0,0,0,0.28)',
      pointerEvents: 'none'
    });

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.remove();
    }, 1600);
  }

  function normalizeUnicode(text) {
    return String(text ?? '')
      .replace(/\r\n?/g, '\n')
      .replace(/[\u200B-\u200D\u2060\uFEFF]/g, '')
      .replace(
        /[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]/g,
        ' '
      );
  }

  function getLatex(element) {
    if (
      element.matches?.(
        'annotation[encoding="application/x-tex"], annotation[encoding="application/tex"]'
      ) &&
      element.textContent?.trim()
    ) {
      return element.textContent.trim();
    }

    const annotation = element.querySelector(
      [
        'annotation[encoding="application/x-tex"]',
        'annotation[encoding="application/tex"]'
      ].join(',')
    );

    if (annotation?.textContent?.trim()) {
      return annotation.textContent.trim();
    }

    for (const attribute of [
      'data-latex',
      'data-tex',
      'data-math',
      'data-formula',
      'data-latex-source',
      'alttext',
      'aria-label',
      'alt',
      'title'
    ]) {
      const value = element.getAttribute(attribute);

      if (value?.trim()) {
        return value.trim();
      }
    }

    return '';
  }

  function normalizeLatexForMarkdown(latex) {
    /*
     * 部分 Markdown/HTML 解析器会把公式中的 < 和 > 当成
     * HTML 标签边界，导致符号丢失或整段公式无法解析。
     * 转成标准 LaTeX 命令后，KaTeX、MathJax 和大多数
     * Markdown 数学插件都能稳定识别。
     */
    return normalizeUnicode(latex)
      .replace(/</g, '\\lt ')
      .replace(/>/g, '\\gt ')
      .trim();
  }

  function getMeaningfulChildNodes(element) {
    return Array.from(element.childNodes).filter(node => {
      if (node.nodeType === Node.TEXT_NODE) {
        return Boolean(node.textContent?.trim());
      }

      return node.nodeType === Node.ELEMENT_NODE;
    });
  }

  function isOnlyMeaningfulChild(element) {
    const parent = element.parentElement;

    if (!parent) {
      return false;
    }

    const meaningfulChildren =
      getMeaningfulChildNodes(parent);

    return (
      meaningfulChildren.length === 1 &&
      meaningfulChildren[0] === element
    );
  }

  function isBlockLikeElement(element) {
    const blockTags = new Set([
      'address',
      'article',
      'aside',
      'blockquote',
      'div',
      'dl',
      'fieldset',
      'figcaption',
      'figure',
      'footer',
      'form',
      'h1',
      'h2',
      'h3',
      'h4',
      'h5',
      'h6',
      'header',
      'hr',
      'li',
      'main',
      'nav',
      'ol',
      'p',
      'pre',
      'section',
      'table',
      'ul'
    ]);

    return blockTags.has(
      element.tagName.toLowerCase()
    );
  }

  function isBlockMath(element) {
    const inlineSelector = [
      '.math-inline',
      '[class*="math-inline"]',
      '[data-math-style="inline"]',
      '[data-display="false"]',
      'math[display="inline"]'
    ].join(',');

    /*
     * 明确标记为 inline 时优先判为行内公式，不能再根据
     * “它是父节点唯一内容”把它误判为块级公式。
     */
    if (
      element.matches(inlineSelector) ||
      (
        element.closest(inlineSelector) &&
        !element.closest('.katex-display, .math-display, [class*="math-display"]')
      )
    ) {
      return false;
    }

    const displaySelector = [
      '.katex-display',
      '.math-display',
      '[class*="math-display"]',
      '[data-math-style="display"]',
      '[data-display="true"]',
      'mjx-container[display="true"]',
      'math[display="block"]'
    ].join(',');

    /*
     * 明确标记为 display 的公式，直接视为块级公式。
     */
    if (
      element.matches(displaySelector) ||
      element.closest(displaySelector)
    ) {
      return true;
    }

    /*
     * 某些渲染器没有 display 标记，但会把公式单独放在
     * 一个块级父元素中。此时也应视为块级公式。
     */
    const parent = element.parentElement;

    if (
      parent &&
      isBlockLikeElement(parent) &&
      isOnlyMeaningfulChild(element)
    ) {
      return true;
    }

    /*
     * 再向上检查最近的段落或块级容器。
     * 只有当该容器的有效内容仅为当前公式时，才判定为块级。
     */
    const blockAncestor = element.closest(
      [
        'p',
        'div',
        'li',
        'blockquote',
        'section',
        'article'
      ].join(',')
    );

    if (
      blockAncestor &&
      blockAncestor !== element
    ) {
      const ancestorText =
        normalizeUnicode(
          blockAncestor.textContent
        ).trim();

      const formulaText =
        normalizeUnicode(
          element.textContent
        ).trim();

      if (
        ancestorText &&
        ancestorText === formulaText
      ) {
        return true;
      }
    }

    return false;
  }

  function getRenderedMathText(element) {
    function read(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        return node.nodeValue || '';
      }

      if (node.nodeType !== Node.ELEMENT_NODE) {
        return '';
      }

      const current = node;
      const tag = current.tagName.toLowerCase();

      if (
        current.getAttribute('aria-hidden') === 'true' &&
        current.matches('.katex-mathml, .MathJax_MathML')
      ) {
        return '';
      }

      const inner = Array.from(current.childNodes)
        .map(read)
        .join('');

      if (tag === 'sub') {
        return `_{${inner.trim()}}`;
      }

      if (tag === 'sup') {
        return `^{${inner.trim()}}`;
      }

      return inner;
    }

    return normalizeUnicode(read(element))
      .replace(/\s+/g, ' ')
      .trim();
  }

  function prepareClone(root) {
    const clone = root.cloneNode(true);

    /*
     * 优先从 KaTeX 的 annotation 读取原始 TeX。
     */
    const annotations = Array.from(
      clone.querySelectorAll(
        [
          'annotation[encoding="application/x-tex"]',
          'annotation[encoding="application/tex"]'
        ].join(',')
      )
    );

    for (const annotation of annotations) {
      if (!annotation.isConnected) {
        continue;
      }

      const latex = normalizeLatexForMarkdown(
        annotation.textContent || ''
      );

      if (!latex) {
        continue;
      }

      const katex = annotation.closest('.katex');
      const display = katex?.closest('.katex-display');
      const math = annotation.closest('math');
      const target = display || katex || math;

      if (!target || !target.isConnected) {
        continue;
      }

      const block = Boolean(display) || isBlockMath(target);
      const markdown = block
        ? `\n\n$$\n${latex}\n$$\n\n`
        : `$${latex}$`;

      target.replaceWith(
        document.createTextNode(markdown)
      );
    }

    /*
     * 删除图片、图片预览和图片附件。
     */
    clone.querySelectorAll(
      [
        'img',
        'picture',
        'figure',
        'canvas',
        '[data-testid*="image"]',
        '[class*="image-container"]',
        '[class*="imageContainer"]'
      ].join(',')
    ).forEach(element => {
      element.remove();
    });

    /*
     * 删除包含图片的附件容器。
     */
    clone.querySelectorAll(
      [
        '[data-testid*="attachment"]',
        '[class*="attachment"]'
      ].join(',')
    ).forEach(element => {
      if (
        element.querySelector(
          'img, picture, figure, canvas'
        )
      ) {
        element.remove();
      }
    });

    const formulaCandidates = Array.from(
      clone.querySelectorAll(
        [
          '.katex-display',
          '.katex',
          'mjx-container',
          'math',
          'annotation[encoding="application/x-tex"]',
          'annotation[encoding="application/tex"]',
          '[data-latex]',
          '[data-tex]',
          '[data-math]',
          '[data-formula]',
          '[data-latex-source]',
          'span[class*="math-inline"]',
          'span[class*="math-display"]',
          'div[class*="math-display"]',
          '[role="math"]'
        ].join(',')
      )
    );

    /*
     * 只处理最外层公式，避免 KaTeX、MathML 被重复读取。
     */
    const outermostFormulas = formulaCandidates.filter(candidate => {
      const formulaAncestor = candidate.parentElement?.closest(
        [
          '.katex-display',
          '.katex',
          'mjx-container',
          'math',
          '[data-latex]',
          '[data-tex]',
          '[data-math]'
        ].join(',')
      );

      return !formulaAncestor;
    });

    for (const element of outermostFormulas) {
      let latex = normalizeLatexForMarkdown(
        getLatex(element)
      );

      if (!latex) {
        latex = normalizeLatexForMarkdown(
          getRenderedMathText(element)
        );
      }

      if (!latex) {
        continue;
      }

      /*
       * 自动区分块级公式和行内公式。
       *
       * 块级公式：
       *
       * $$公式$$
       *
       * 行内公式：
       *
       * 正文 $公式$ 正文
       */
      const markdown = isBlockMath(element)
        ? `\n\n$$\n${latex}\n$$\n\n`
        : `$${latex}$`;

      element.replaceWith(
        document.createTextNode(markdown)
      );
    }

    /*
     * 删除操作按钮、图标、脚本和本脚本按钮。
     */
    clone.querySelectorAll(
      [
        'button',
        'svg',
        'style',
        'script',
        'noscript',
        '[data-testid*="copy"]',
        '[aria-label*="Copy"]',
        '[aria-label*="copy"]',
        '[aria-label*="复制"]',
        `.${BUTTON_CLASS}`,
        `.${WRAPPER_CLASS}`
      ].join(',')
    ).forEach(element => {
      element.remove();
    });

    return clone;
  }

  function getListDepth(listItem) {
    let depth = 0;
    let current = listItem.parentElement;

    while (current) {
      if (current.matches('ul, ol')) {
        depth++;
      }

      current = current.parentElement;
    }

    return Math.max(0, depth - 1);
  }

  function getListMarker(listItem) {
    const list = listItem.parentElement;

    if (
      !list ||
      list.tagName.toLowerCase() !== 'ol'
    ) {
      return '-';
    }

    const items = Array.from(list.children).filter(child => {
      return child.tagName.toLowerCase() === 'li';
    });

    const start = Number(
      list.getAttribute('start') || 1
    );

    const index = Math.max(
      items.indexOf(listItem),
      0
    );

    return `${start + index}.`;
  }

  function getCodeFence(code) {
    const matches = Array.from(
      code.matchAll(/`+/g),
      match => match[0].length
    );

    const maximum = matches.length
      ? Math.max(...matches)
      : 0;

    return '`'.repeat(
      Math.max(3, maximum + 1)
    );
  }

  function escapeTableCell(text) {
    return normalizeUnicode(text)
      .replace(/\|/g, '\\|')
      .replace(/\n+/g, '<br>')
      .trim();
  }

  function cleanListItemText(text) {
    return normalizeUnicode(text)
      .split('\n')
      .map(line => line.trim())
      .filter(Boolean)
      .join('\n')
      .trim();
  }

  function isTableLine(line) {
    return /^\s*\|.*\|\s*$/.test(line);
  }

  function isFenceLine(line) {
    return /^\s*(`{3,}|~{3,})/.test(line);
  }

  function isHorizontalRuleLine(line) {
    return /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/.test(line);
  }

  function tableToMarkdown(table, walk) {
    /*
     * 只读取当前 table 的直接行，避免嵌套表格造成重复。
     */
    let rows = Array.from(
      table.querySelectorAll(
        [
          ':scope > thead > tr',
          ':scope > tbody > tr',
          ':scope > tfoot > tr',
          ':scope > tr'
        ].join(',')
      )
    );

    /*
     * 某些 DOM 不支持上面的 :scope 组合时兜底。
     */
    if (!rows.length) {
      rows = Array.from(
        table.querySelectorAll('tr')
      );
    }

    if (!rows.length) {
      return '';
    }

    const matrix = rows
      .map(row => {
        return Array.from(
          row.querySelectorAll(
            ':scope > th, :scope > td'
          )
        ).map(cell => {
          const value = Array.from(cell.childNodes)
            .map(child => {
              return walk(
                child,
                cell.tagName.toLowerCase()
              );
            })
            .join('');

          return escapeTableCell(value);
        });
      })
      .filter(row => row.length > 0);

    if (!matrix.length) {
      return '';
    }

    const columnCount = Math.max(
      ...matrix.map(row => row.length)
    );

    for (const row of matrix) {
      while (row.length < columnCount) {
        row.push('');
      }
    }

    const header = matrix[0];
    const body = matrix.slice(1);

    const lines = [
      `| ${header.join(' | ')} |`,
      `| ${header.map(() => '---').join(' | ')} |`,
      ...body.map(row => {
        return `| ${row.join(' | ')} |`;
      })
    ];

    /*
     * 表格前后必须有空白行。
     *
     * 后续 cleanMarkdown 会：
     * - 保留表格前的一个空白行；
     * - 保留表格后的一个空白行；
     * - 删除其他不必要空白行。
     */
    return `\n\n${lines.join('\n')}\n\n`;
  }

  function removeMarkdownImages(markdown) {
    let result = markdown;

    /*
     * 删除普通图片：
     *
     * ![alt](url)
     * ![](url)
     *
     * 支持 URL 内有一层括号的常见情况。
     */
    result = result.replace(
      /!\[[^\]]*]\((?:\\.|[^()\n]|\([^()\n]*\))*\)/g,
      ''
    );

    /*
     * 删除引用式图片：
     *
     * ![alt][id]
     */
    result = result.replace(
      /!\[[^\]]*]\[[^\]]*]/g,
      ''
    );

    /*
     * 删除简写引用图片：
     *
     * ![alt][]
     */
    result = result.replace(
      /!\[[^\]]*]\[\]/g,
      ''
    );

    return result;
  }

  function findPreviousNonBlank(lines, index) {
    for (let i = index - 1; i >= 0; i--) {
      if (lines[i] !== '') {
        return lines[i];
      }
    }

    return '';
  }

  function findNextNonBlank(lines, index) {
    for (let i = index + 1; i < lines.length; i++) {
      if (lines[i] !== '') {
        return lines[i];
      }
    }

    return '';
  }

  function compactMathBlocks(lines) {
    const output = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      /*
       * 支持以下形式：
       *
       * $$
       * a+b
       * =
       * c
       * $$
       *
       * 转换为：
       *
       * $$a+b
       * =
       * c$$
       */
      if (line === '$$') {
        const formulaLines = [];
        let closingIndex = -1;

        for (let j = i + 1; j < lines.length; j++) {
          if (lines[j] === '$$') {
            closingIndex = j;
            break;
          }

          formulaLines.push(lines[j]);
        }

        if (closingIndex !== -1) {
          output.push(
            `$$${formulaLines.join('\n')}$$`
          );

          i = closingIndex;
          continue;
        }
      }

      output.push(line);
    }

    return output;
  }

  function promoteStandaloneInlineMath(lines) {
    const output = [];
    let insideFence = false;
    let fenceCharacter = '';

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const fenceMatch = line.match(/^\s*((?:\x60){3,}|~{3,})/);

      if (fenceMatch) {
        const character = fenceMatch[1][0];

        if (!insideFence) {
          insideFence = true;
          fenceCharacter = character;
        } else if (character === fenceCharacter) {
          insideFence = false;
          fenceCharacter = '';
        }

        output.push(line);
        continue;
      }

      if (
        !insideFence &&
        /^\$(?!\$).+\$$/.test(line)
      ) {
        output.push(`$$${line.slice(1, -1)}$$`);
        continue;
      }

      if (
        !insideFence &&
        /^\$(?!\$)/.test(line) &&
        !/\$/.test(line)
      ) {
        const firstLine = line.slice(1);
        const formulaLines = [firstLine];
        let closingIndex = -1;

        /*
         * 使用花括号深度识别多行公式结尾。这样 \boxed{ ... }
         * 内部即使出现 $GRPO=$，也不会被误认为外层公式结束。
         */
        let braceDepth =
          (firstLine.match(/(?<!\\){/g) || []).length -
          (firstLine.match(/(?<!\\)}/g) || []).length;

        for (let j = i + 1; j < lines.length; j++) {
          const current = lines[j];
          const opens =
            (current.match(/(?<!\\){/g) || []).length;
          const closes =
            (current.match(/(?<!\\)}/g) || []).length;

          const isBalancedClosingLine =
            /\$/.test(current) &&
            braceDepth + opens - closes <= 0;

          if (isBalancedClosingLine) {
            formulaLines.push(
              current.slice(0, -1)
            );
            closingIndex = j;
            break;
          }

          /*
           * 清除块级公式内部错误嵌套的单美元定界符：
           * $GRPO=$ -> GRPO=
           */
          formulaLines.push(
            current.replace(/^\$(?!\$)/, '')
              .replace(/(?<!\$)\$/, '')
          );

          braceDepth += opens - closes;
        }

        if (closingIndex !== -1) {
          output.push(
            `$${formulaLines.join('\n')}$`
          );
          i = closingIndex;
          continue;
        }
      }

      output.push(line);
    }

    return output;
  }

  function repairMalformedNestedMath(text) {
    const lines = text.split('\n');
    const output = [];
    let insideFence = false;
    let fenceCharacter = '';

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const fenceMatch = line.match(/^\s*((?:\x60){3,}|~{3,})/);

      if (fenceMatch) {
        const character = fenceMatch[1][0];

        if (!insideFence) {
          insideFence = true;
          fenceCharacter = character;
        } else if (character === fenceCharacter) {
          insideFence = false;
          fenceCharacter = '';
        }

        output.push(line);
        continue;
      }

      /*
       * 修复：
       * $\\boxed{
       * $$GRPO=$$
       * ...
       * }$
       *
       * 重建为一个且仅一个块级公式。
       */
      if (
        !insideFence &&
        /^\$(?!\$).*(?<!\\){\s*$/.test(line)
      ) {
        const formulaLines = [line.slice(1)];
        let braceDepth =
          (line.match(/(?<!\\){/g) || []).length -
          (line.match(/(?<!\\)}/g) || []).length;
        let closingIndex = -1;

        for (let j = i + 1; j < lines.length; j++) {
          const current = lines[j];
          const withoutDelimiters = current
            .replace(/^\$\$(.*)\$\$$/, '$1')
            .replace(/^\$(?!\$)(.*)\$$/, '$1');

          const opens =
            (withoutDelimiters.match(/(?<!\\){/g) || []).length;
          const closes =
            (withoutDelimiters.match(/(?<!\\)}/g) || []).length;

          const closesOuter =
            /\$$/.test(current) &&
            braceDepth + opens - closes <= 0;

          if (closesOuter) {
            formulaLines.push(
              current.replace(/\$$/, '')
            );
            closingIndex = j;
            break;
          }

          formulaLines.push(withoutDelimiters);
          braceDepth += opens - closes;
        }

        if (closingIndex !== -1) {
          output.push(
            `$$${formulaLines.join('\n')}$$`
          );
          i = closingIndex;
          continue;
        }
      }

      /*
       * 普通跨行单美元公式：
       *
       * $L
       * =
       * R$
       *
       * 必须重建为块级 $$...$$。
       */
      if (
        !insideFence &&
        /^\$(?!\$)/.test(line) &&
        !/\$$/.test(line)
      ) {
        const formulaLines = [line.slice(1)];
        let closingIndex = -1;

        for (
          let j = i + 1;
          j < lines.length && j <= i + 100;
          j++
        ) {
          const current = lines[j];

          if (/\$$/.test(current)) {
            formulaLines.push(
              current.slice(0, -1)
            );
            closingIndex = j;
            break;
          }

          formulaLines.push(
            current
              .replace(/^\$\$(.*)\$\$$/, '$1')
              .replace(/^\$(?!\$)(.*)\$$/, '$1')
          );
        }

        if (closingIndex !== -1) {
          output.push(
            `$$${formulaLines.join('\n')}$$`
          );
          i = closingIndex;
          continue;
        }
      }

      output.push(line);
    }

    return output.join('\n');
  }

  function wrapPlainMathLines(lines) {
    const output = [];
    let insideFence = false;
    let fenceCharacter = '';

    for (const line of lines) {
      const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);

      if (fenceMatch) {
        const character = fenceMatch[1][0];

        if (!insideFence) {
          insideFence = true;
          fenceCharacter = character;
        } else if (character === fenceCharacter) {
          insideFence = false;
          fenceCharacter = '';
        }

        output.push(line);
        continue;
      }

      if (
        !insideFence &&
        line &&
        !line.includes('$') &&
        !/\s/.test(line) &&
        !/[\u3400-\u9FFF]/.test(line) &&
        !/^(?:[#>|`]|[-*+]\s|\d+\.\s|https?:\/\/)/.test(line) &&
        /[=≈≃≠≤≥<>∼~∑∏∫√∞±×÷⋅·∣|]/.test(line) &&
        /[A-Za-z0-9α-ωΑ-Ωθπλϵετ]/.test(line)
      ) {
        output.push(`$$${line}$$`);
        continue;
      }

      output.push(line);
    }

    return output;
  }

  function cleanMarkdown(markdown) {
    let text = normalizeUnicode(markdown);
    text = removeMarkdownImages(text);

    const rawLines = text.split('\n');
    const normalizedLines = [];

    let insideFence = false;
    let fenceCharacter = '';

    /*
     * 第一遍：
     * - 保留代码块内部格式；
     * - 删除行尾空格；
     * - 非代码块内容 trim；
     * - 空白行统一变成空字符串。
     */
    for (const rawLine of rawLines) {
      const rightTrimmed = rawLine.replace(/[ \t]+$/g, '');
      const fenceMatch = rightTrimmed.match(
        /^\s*(`{3,}|~{3,})/
      );

      if (fenceMatch) {
        const currentCharacter = fenceMatch[1][0];

        if (!insideFence) {
          insideFence = true;
          fenceCharacter = currentCharacter;
          normalizedLines.push(
            rightTrimmed.trim()
          );
          continue;
        }

        if (currentCharacter === fenceCharacter) {
          normalizedLines.push(
            rightTrimmed.trim()
          );

          insideFence = false;
          fenceCharacter = '';
          continue;
        }
      }

      if (insideFence) {
        normalizedLines.push(rightTrimmed);
        continue;
      }

      normalizedLines.push(
        rightTrimmed.trim()
      );
    }

    /*
     * 第二遍：
     * 保留表格和水平分隔线前后的必要空行。
     *
     * 普通标题、正文、公式、列表、代码块之间不留空行。
     */
    const compactedLines = [];

    insideFence = false;
    fenceCharacter = '';

    for (let i = 0; i < normalizedLines.length; i++) {
      const line = normalizedLines[i];

      const fenceMatch = line.match(
        /^\s*(`{3,}|~{3,})/
      );

      if (fenceMatch) {
        const currentCharacter = fenceMatch[1][0];

        if (!insideFence) {
          insideFence = true;
          fenceCharacter = currentCharacter;
          compactedLines.push(line);
          continue;
        }

        if (currentCharacter === fenceCharacter) {
          compactedLines.push(line);
          insideFence = false;
          fenceCharacter = '';
          continue;
        }
      }

      if (insideFence) {
        compactedLines.push(line);
        continue;
      }

      if (line !== '') {
        compactedLines.push(line);
        continue;
      }

      const previous = findPreviousNonBlank(
        normalizedLines,
        i
      );

      const next = findNextNonBlank(
        normalizedLines,
        i
      );

      /*
       * Markdown 表格前保留一个空白行：
       *
       * 正文
       *
       * | 表格 |
       */
      const blankBeforeTable =
        !isTableLine(previous) &&
        isTableLine(next);

      /*
       * Markdown 表格后保留一个空白行：
       *
       * | 表格 |
       *
       * 正文
       */
      const blankAfterTable =
        isTableLine(previous) &&
        !isTableLine(next);

      /*
       * 水平分隔线前后必须保留空白行，否则：
       *
       * 正文
       * ---
       *
       * 可能会被 Markdown 解析为 Setext 标题，
       * 而不是水平分隔线。
       */
      const blankBeforeHorizontalRule =
        !isHorizontalRuleLine(previous) &&
        isHorizontalRuleLine(next);

      const blankAfterHorizontalRule =
        isHorizontalRuleLine(previous) &&
        !isHorizontalRuleLine(next);

      if (
        blankBeforeTable ||
        blankAfterTable ||
        blankBeforeHorizontalRule ||
        blankAfterHorizontalRule
      ) {
        if (
          compactedLines.length > 0 &&
          compactedLines[
            compactedLines.length - 1
          ] !== ''
        ) {
          compactedLines.push('');
        }
      }
    }

    /*
     * 第三遍：
     * 删除连续空行，但保留表格和水平分隔线前后的单个空白行。
     */
    const deduplicatedLines = [];

    for (const line of compactedLines) {
      if (
        line === '' &&
        deduplicatedLines[
          deduplicatedLines.length - 1
        ] === ''
      ) {
        continue;
      }

      deduplicatedLines.push(line);
    }

    /*
     * 清理首尾空行。
     */
    while (
      deduplicatedLines.length > 0 &&
      deduplicatedLines[0] === ''
    ) {
      deduplicatedLines.shift();
    }

    while (
      deduplicatedLines.length > 0 &&
      deduplicatedLines[
        deduplicatedLines.length - 1
      ] === ''
    ) {
      deduplicatedLines.pop();
    }

    const mathCompacted = compactMathBlocks(
      deduplicatedLines
    );

    const mathWrapped = wrapPlainMathLines(
      mathCompacted
    );

    const mathClassified =
      promoteStandaloneInlineMath(
        mathWrapped
      );

    const joined = mathClassified
      .join('\n')
      .replace(/\n{3,}/g, '\n\n');

    return repairMalformedNestedMath(joined)
      .trim();
  }

  function htmlToMarkdown(root) {
    const clone = prepareClone(root);

    function walk(node, parentTag = '') {
      if (node.nodeType === Node.TEXT_NODE) {
        return node.nodeValue || '';
      }

      if (node.nodeType !== Node.ELEMENT_NODE) {
        return '';
      }

      const element = node;
      const tag = element.tagName.toLowerCase();

      /*
       * 图片直接删除，不生成 Markdown 图片链接。
       */
      if (
        tag === 'img' ||
        tag === 'picture' ||
        tag === 'figure' ||
        tag === 'canvas'
      ) {
        return '';
      }

      /*
       * 代码块必须提前读取。
       */
      if (tag === 'pre') {
        const codeElement =
          element.querySelector('code');

        const code =
          codeElement?.textContent ??
          element.textContent ??
          '';

        const className =
          codeElement?.className ||
          '';

        const language =
          className.match(
            /language-([\w#+.-]+)/
          )?.[1] ||
          element.getAttribute('data-language') ||
          '';

        const fence = getCodeFence(code);

        return [
          `${fence}${language}`,
          code.replace(/\n$/, ''),
          fence
        ].join('\n') + '\n';
      }

      if (tag === 'table') {
        return tableToMarkdown(
          element,
          walk
        );
      }

      const inner = Array.from(element.childNodes)
        .map(child => {
          return walk(child, tag);
        })
        .join('');

      switch (tag) {
        case 'br':
          return '\n';

        case 'p': {
          const content = inner.trim();

          if (!content) {
            return '';
          }

          /*
           * li 内的 p 不额外添加段落换行。
           */
          if (parentTag === 'li') {
            return content;
          }

          return `${content}\n`;
        }

        case 'h1':
        case 'h2':
        case 'h3':
        case 'h4':
        case 'h5':
        case 'h6': {
          const content = inner.trim();

          if (!content) {
            return '';
          }

          const level = Number(tag[1]);

          return (
            `${'#'.repeat(level)} ` +
            `${content}\n`
          );
        }

        case 'strong':
        case 'b': {
          const content = inner.trim();

          return content
            ? `**${content}**`
            : '';
        }

        case 'em':
        case 'i': {
          const content = inner.trim();

          return content
            ? `*${content}*`
            : '';
        }

        case 'del':
        case 's': {
          const content = inner.trim();

          return content
            ? `~~${content}~~`
            : '';
        }

        case 'code': {
          if (parentTag === 'pre') {
            return inner;
          }

          const content = inner.trim();

          if (!content) {
            return '';
          }

          const matches = Array.from(
            content.matchAll(/`+/g),
            match => match[0].length
          );

          const maximum = matches.length
            ? Math.max(...matches)
            : 0;

          const fence = '`'.repeat(
            Math.max(1, maximum + 1)
          );

          return `${fence}${content}${fence}`;
        }

        case 'blockquote': {
          const content = inner.trim();

          if (!content) {
            return '';
          }

          return (
            content
              .split('\n')
              .filter(Boolean)
              .map(line => {
                return `> ${line.trim()}`;
              })
              .join('\n') +
            '\n'
          );
        }

        case 'ul':
        case 'ol': {
          const content = inner.trim();

          if (!content) {
            return '';
          }

          return `${content}\n`;
        }

        case 'li': {
          const depth = getListDepth(element);
          const indent = '  '.repeat(depth);
          const marker = getListMarker(element);

          const content =
            cleanListItemText(inner);

          if (!content) {
            return '';
          }

          const lines = content.split('\n');

          const firstLine =
            `${indent}${marker} ${lines[0]}`;

          const continuationIndent =
            `${indent}${' '.repeat(
              marker.length + 1
            )}`;

          const remainingLines = lines
            .slice(1)
            .map(line => {
              return (
                `${continuationIndent}` +
                `${line}`
              );
            });

          return (
            `${[
              firstLine,
              ...remainingLines
            ].join('\n')}\n`
          );
        }

        case 'a': {
          const href =
            element.getAttribute('href');

          const text = inner.trim();

          /*
           * 空链接通常是图片的包裹链接。
           */
          if (!text) {
            return '';
          }

          if (!href) {
            return text;
          }

          if (
            href.startsWith('data:image') ||
            href.startsWith('blob:')
          ) {
            return '';
          }

          if (text === href) {
            return href;
          }

          return `[${text}](${href})`;
        }

        case 'img':
        case 'picture':
        case 'figure':
        case 'canvas':
          return '';

        case 'hr':
          /*
           * 水平线前后强制留出空白行。
           * cleanMarkdown 会将其压缩为恰好一个空白行。
           */
          return '\n\n---\n\n';

        case 'thead':
        case 'tbody':
        case 'tfoot':
        case 'tr':
        case 'th':
        case 'td':
          return inner;

        default:
          return inner;
      }
    }

    return cleanMarkdown(
      walk(clone)
    );
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (clipboardError) {
      try {
        GM_setClipboard(text, 'text');
        return true;
      } catch (tampermonkeyError) {
        console.error(
          '[Compact Markdown Copier] Clipboard error:',
          clipboardError,
          tampermonkeyError
        );

        return false;
      }
    }
  }

  function getAssistantMessages() {
    const messages = new Set();

    document
      .querySelectorAll(
        [
          '[data-message-author-role="assistant"]',
          '[data-turn="assistant"]',
          '[data-testid="assistant-message"]',
          'article[data-turn="assistant"]'
        ].join(',')
      )
      .forEach(element => {
        messages.add(element);
      });

    document
      .querySelectorAll(
        [
          'article[data-testid^="conversation-turn"]',
          'main article'
        ].join(',')
      )
      .forEach(article => {
        const assistant = article.querySelector(
          [
            '[data-message-author-role="assistant"]',
            '[data-turn="assistant"]',
            '[data-testid="assistant-message"]'
          ].join(',')
        );

        if (assistant) {
          messages.add(assistant);
          return;
        }

        const markdown = article.querySelector(
          [
            '.markdown',
            '[class*="markdown"]',
            '.prose'
          ].join(',')
        );

        const hasButton =
          article.querySelector('button');

        const text =
          article.innerText?.trim();

        if (
          markdown &&
          hasButton &&
          text
        ) {
          messages.add(article);
        }
      });

    return Array.from(messages);
  }

  function getContentRoot(message) {
    return (
      message.querySelector('.markdown') ||
      message.querySelector(
        '[class*="markdown"]'
      ) ||
      message.querySelector('.prose') ||
      message.querySelector(
        '[data-message-content]'
      ) ||
      message
    );
  }

  function findActionBar(message) {
    const candidates = Array.from(
      message.querySelectorAll('div')
    ).filter(container => {
      const directButtons =
        container.querySelectorAll(
          ':scope > button'
        );

      const nestedButtons =
        container.querySelectorAll(
          ':scope > div > button'
        );

      return (
        directButtons.length +
        nestedButtons.length >= 2
      );
    });

    return candidates.at(-1) || null;
  }

  function createCopyButton(content) {
    const button =
      document.createElement('button');

    button.type = 'button';
    button.className = BUTTON_CLASS;
    button.textContent = '复制 Markdown';
    button.title =
      '复制紧凑 Markdown；自动区分 $ 与 $$；表格和横线正确隔离';

    Object.assign(button.style, {
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      marginLeft: '6px',
      padding: '4px 8px',
      border:
        '1px solid rgba(128,128,128,0.35)',
      borderRadius: '7px',
      background: 'transparent',
      color: 'inherit',
      cursor: 'pointer',
      fontSize: '12px',
      lineHeight: '1.4'
    });

    button.addEventListener(
      'mouseenter',
      () => {
        button.style.background =
          'rgba(128,128,128,0.12)';
      }
    );

    button.addEventListener(
      'mouseleave',
      () => {
        button.style.background =
          'transparent';
      }
    );

    button.addEventListener(
      'click',
      async event => {
        event.preventDefault();
        event.stopPropagation();

        const markdown =
          htmlToMarkdown(content);

        if (!markdown) {
          showToast(
            '未提取到 Markdown 内容'
          );

          return;
        }

        const success =
          await copyText(markdown);

        if (!success) {
          showToast('剪贴板写入失败');
          return;
        }

        const previousText =
          button.textContent;

        button.textContent = '已复制';

        showToast(
          '已复制紧凑 Markdown'
        );

        setTimeout(() => {
          button.textContent =
            previousText;
        }, 1200);
      }
    );

    return button;
  }

  function addButtonToMessage(message) {
    if (!(message instanceof Element)) {
      return;
    }

    if (
      message.querySelector(
        `.${BUTTON_CLASS}`
      ) ||
      message.parentElement?.querySelector(
        `:scope > .${WRAPPER_CLASS}`
      )
    ) {
      return;
    }

    const content =
      getContentRoot(message);

    if (!content) {
      return;
    }

    const button =
      createCopyButton(content);

    const wrapper =
      document.createElement('div');

    wrapper.className = WRAPPER_CLASS;

    Object.assign(wrapper.style, {
      display: 'flex',
      alignItems: 'center',
      position: 'relative',
      visibility: 'visible',
      opacity: '1',
      zIndex: '10',
      marginTop: '8px'
    });

    wrapper.appendChild(button);

    content.insertAdjacentElement(
      'afterend',
      wrapper
    );
  }

  /*
   * 新版 ChatGPT 的消息节点属性可能随界面更新而变化，但每条
   * 助手回复下方的原生“复制”按钮相对稳定。优先从原生复制按钮
   * 反向定位整条回复，再把自定义按钮放进同一个操作栏。
   */
  function addButtonsFromNativeCopyActions() {
    const nativeCopyButtons = document.querySelectorAll(
      [
        'button[data-testid="copy-turn-action-button"]',
        'button[data-testid*="copy-turn"]',
        'button[aria-label="复制"]',
        'button[aria-label="Copy"]'
      ].join(',')
    );

    for (const nativeButton of nativeCopyButtons) {
      if (
        nativeButton.classList.contains(BUTTON_CLASS) ||
        nativeButton.dataset.tmMarkdownEnhanced === 'true'
      ) {
        continue;
      }

      const turn = nativeButton.closest(
        [
          '[data-testid^="conversation-turn"]',
          '[data-turn="assistant"]',
          'article'
        ].join(',')
      );

      if (!turn) {
        continue;
      }

      const content = getContentRoot(turn);

      if (!content) {
        continue;
      }

      const actionBar =
        nativeButton.closest(
          '[data-testid="turn-action-buttons"]'
        ) || nativeButton.parentElement;

      if (!actionBar) {
        continue;
      }

      if (actionBar.querySelector(`.${BUTTON_CLASS}`)) {
        nativeButton.dataset.tmMarkdownEnhanced = 'true';
        continue;
      }

      const button = createCopyButton(content);

      Object.assign(button.style, {
        display: 'inline-flex',
        visibility: 'visible',
        opacity: '1',
        flexShrink: '0'
      });

      nativeButton.insertAdjacentElement(
        'afterend',
        button
      );

      nativeButton.dataset.tmMarkdownEnhanced = 'true';
    }
  }

  function getLastAssistantContent() {
    const messages = getAssistantMessages();

    if (messages.length) {
      return getContentRoot(
        messages[messages.length - 1]
      );
    }

    /*
     * 新版界面属性变化时的最终兜底。ChatGPT 的助手回复正文
     * 通常仍使用 markdown 或 prose 容器，取页面中最后一个。
     */
    const candidates = Array.from(
      document.querySelectorAll(
        [
          'main .markdown',
          'main [class*="markdown"]',
          'main .prose',
          '[data-message-content]'
        ].join(',')
      )
    ).filter(element => {
      return Boolean(element.innerText?.trim());
    });

    return candidates.at(-1) || null;
  }

  function ensureGlobalCopyButton() {
    if (document.getElementById(GLOBAL_BUTTON_ID)) {
      return;
    }

    const button = document.createElement('button');

    button.id = GLOBAL_BUTTON_ID;
    button.type = 'button';
    button.textContent = '复制最后回复 Markdown';
    button.title = '复制当前页面最后一条 ChatGPT 回复的 Markdown';

    Object.assign(button.style, {
      position: 'fixed',
      right: '20px',
      bottom: '76px',
      zIndex: '2147483646',
      display: 'inline-flex',
      alignItems: 'center',
      justifyContent: 'center',
      visibility: 'visible',
      opacity: '1',
      padding: '9px 13px',
      border: '1px solid rgba(128,128,128,0.42)',
      borderRadius: '9px',
      background: '#202123',
      color: '#ffffff',
      boxShadow: '0 4px 16px rgba(0,0,0,0.22)',
      cursor: 'pointer',
      fontSize: '13px',
      fontWeight: '500',
      lineHeight: '1.4'
    });

    button.addEventListener('click', async event => {
      event.preventDefault();
      event.stopPropagation();

      const content = getLastAssistantContent();

      if (!content) {
        showToast('未找到 ChatGPT 回复正文');
        return;
      }

      const markdown = htmlToMarkdown(content);

      if (!markdown) {
        showToast('未提取到 Markdown 内容');
        return;
      }

      const success = await copyText(markdown);

      if (!success) {
        showToast('剪贴板写入失败');
        return;
      }

      const previousText = button.textContent;
      button.textContent = '已复制';
      showToast('已复制最后回复 Markdown');

      setTimeout(() => {
        button.textContent = previousText;
      }, 1200);
    });

    document.body.appendChild(button);
  }

  let scanTimer = null;

  function scheduleScan(delay = 80) {
    clearTimeout(scanTimer);

    scanTimer = setTimeout(() => {
      document.getElementById(GLOBAL_BUTTON_ID)?.remove();
      document
        .querySelectorAll('.' + WRAPPER_CLASS)
        .forEach(element => element.remove());

      addButtonsFromNativeCopyActions();
    }, delay);
  }

  const observer =
    new MutationObserver(() => {
      scheduleScan(50);
    });

  observer.observe(
    document.documentElement,
    {
      childList: true,
      subtree: true
    }
  );

  window.addEventListener(
    'scroll',
    () => {
      scheduleScan(100);
    },
    true
  );

  window.addEventListener(
    'resize',
    () => {
      scheduleScan(100);
    }
  );

  document.addEventListener(
    'visibilitychange',
    () => {
      if (!document.hidden) {
        scheduleScan(100);
      }
    }
  );

  /*
   * 兜底扫描，处理历史消息虚拟化加载。
   */
  setInterval(() => {
    scheduleScan(0);
  }, 1500);

  scheduleScan(0);

  console.info(
    '[ChatGPT Compact Markdown Copier] v5.3.0 loaded'
  );
})();

```

复制chatgpt内容的时候公式会存在问题，因为它使用的是`[]`，这里使用tampermonkey并结合脚本，添加一个复制按钮，对公式进行解析，以便markdown能够正常解析。
