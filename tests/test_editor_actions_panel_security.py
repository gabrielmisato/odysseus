import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_action_names_and_unknown_steps_are_literal_text():
    script = r'''
      const fs = require('fs');
      const { chromium } = require('playwright');
      (async () => {
        const browser = await chromium.launch({ headless: true });
        try {
          const page = await browser.newPage();
          await page.route('http://actions.test/**', route => route.fulfill({
            status: 200, contentType: 'text/html', body: '<html><body></body></html>',
          }));
          await page.goto('http://actions.test/');
          const source = fs.readFileSync(process.argv[1], 'utf8');
          const moduleUrl = 'data:text/javascript;base64,' + Buffer.from(source).toString('base64');
          const result = await page.evaluate(async url => {
            const attack = '<img src=x onerror="window.__actionsXss=true">';
            localStorage.setItem('ge-actions', JSON.stringify({ [attack]: ['flip-h'] }));
            const { createActionsPanel } = await import(url);
            const panel = createActionsPanel();
            document.body.append(panel.element);
            panel.setSteps([attack]);
            return {
              images: panel.element.querySelectorAll('img').length,
              saved: panel.element.querySelector('.ge-actions-saved-row button').textContent,
              step: panel.element.querySelector('.ge-actions-step-label').textContent,
              executed: Boolean(window.__actionsXss),
              attack,
            };
          }, moduleUrl);
          console.log(JSON.stringify(result));
        } finally { await browser.close(); }
      })().catch(error => { console.error(error); process.exit(1); });
    '''
    result = subprocess.run(
        ['node', '-e', script, str(ROOT / 'static/js/editor/actions-panel.js')],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed['images'] == 0
    assert observed['saved'] == observed['step'] == observed['attack']
    assert not observed['executed']
