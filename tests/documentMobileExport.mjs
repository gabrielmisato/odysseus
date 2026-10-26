import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const browser = await chromium.launch({ headless: true });
try {
  for (const width of [390, 900]) {
    const page = await browser.newPage({ viewport: { width, height: 844 }, acceptDownloads: true });
    await page.goto('http://127.0.0.1:7011/static/js/documentStats.js');
    await page.setContent('<link rel="stylesheet" href="/static/css/00-tokens.css"><div id="toast"></div><div id="chat-container"></div><div id="sidebar"></div>');
    await page.evaluate(async () => {
      const mod = await import('/static/js/document.js?mobile-export-regression=1');
      mod.init('/api');
      mod.injectFreshDoc({ id: 'export-test', title: 'Export test', language: 'markdown', version_count: 1, current_content: '# Mobile export\n\nActual document content.' });
    });
    await page.locator('#doc-footer-export-btn').waitFor({ state: 'visible' });
    await page.evaluate(() => {
      document.querySelector('.doc-editor-pane').style.setProperty('z-index', '12000', 'important');
    });
    await page.locator('#doc-footer-export-btn').click();
    const menu = page.locator('#doc-export-menu');
    await menu.waitFor({ state: 'visible' });
    const bounds = await menu.boundingBox();
    assert(bounds.x >= 0 && bounds.y >= 0);
    assert(bounds.x + bounds.width <= width && bounds.y + bounds.height <= 844);
    assert(await menu.evaluate(el => Number(getComputedStyle(el).zIndex) > 12000));
    await page.screenshot({ path: `/tmp/document-export-${width}.png` });
    // A second click closes the menu, then reopening must still allow export.
    await page.locator('#doc-footer-export-btn').click();
    assert.equal(await menu.count(), 0);
    await page.locator('#doc-footer-export-btn').click();
    const pending = page.waitForEvent('download');
    await page.locator('#doc-export-menu button').filter({ hasText: /^Export Markdown$/ }).click();
    const download = await pending;
    assert(download.suggestedFilename().endsWith('.md'));
    assert.equal(await download.failure(), null);
    const stream = await download.createReadStream();
    let content = '';
    for await (const chunk of stream) content += chunk;
    assert.equal(content, '# Mobile export\n\nActual document content.');
    await page.close();
    console.log(`PASS ${width}px: visible menu, toggle, download contents`);
  }
} finally {
  await browser.close();
}
