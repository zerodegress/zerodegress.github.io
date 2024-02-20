import js from '@eslint/js'
import prettier from 'eslint-config-prettier'
import typescript from 'typescript-eslint'
import react from 'eslint-plugin-react/configs/jsx-runtime.js'
import { FlatCompat } from '@eslint/eslintrc'

const compat = new FlatCompat({
  recommendedConfig: js.configs.recommended,
})

/** @type {import('eslint').Linter.FlatConfig[]} */
export default [
  {
    ignores: ['.astro/*', '**/env.d.ts'],
  },
  js.configs.recommended,
  ...compat.config({
    plugins: ['react-refresh'],
    extends: ['plugin:react-hooks/recommended'],
    rules: {
      'react-refresh/only-export-components': 'warn',
    },
  }),
  ...typescript.configs.recommended,
  react,
  prettier,
]
