import js from '@eslint/js'
import globals from 'globals'
import react from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'

// Flat config. The project is React 18 in plain JavaScript (no TypeScript),
// so this is the whole lint surface: core JS correctness plus the two React
// rules that catch real bugs (hook order, non-component exports breaking HMR).
export default [
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: 'module',
      globals: { ...globals.browser },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    plugins: {
      react,
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      // Without these two, core no-unused-vars does not see identifiers that
      // are only referenced inside JSX, and flags every component import.
      'react/jsx-uses-react': 'error',
      'react/jsx-uses-vars': 'error',
      ...reactHooks.configs.recommended.rules,
      // Fetch-on-mount (`useEffect(() => { load(0) }, [load])`) trips this
      // rule because `load` sets a loading flag before its first await. The
      // flag already initialises to true, so that write is a no-op on mount,
      // and deferring it would only risk a double fetch. Kept visible as a
      // warning instead of rewritten.
      'react-hooks/set-state-in-effect': 'warn',
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // Unused args are often there for signature clarity; ignore the
      // underscore convention and caught-error bindings.
      'no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', caughtErrors: 'none' },
      ],
    },
  },
  {
    // Vite config runs in Node, not the browser.
    files: ['vite.config.js', 'eslint.config.js'],
    languageOptions: { globals: { ...globals.node } },
  },
]
