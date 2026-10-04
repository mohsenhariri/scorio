---
title: Installation and imports
---

## Install from npm

```sh
npm install scorio
```

The package declares Node.js 18 or later and ships ES modules, CommonJS, and
TypeScript declarations. It has no runtime dependencies. For a browser app,
install it through your package manager and import it through your bundler.

Calculations are synchronous. Run long ranking fits or posterior calculations
in a Web Worker if your browser app needs to remain responsive during them.

Use a subpath when you need one part of the package:

```ts
import { bayes, passAtK } from "scorio/eval";

console.log(bayes([0, 1, 1]));
console.log(passAtK([0, 1, 1], 2));
```

Or import namespaces from the root:

```js
import { eval as metrics, rank, aggregate } from "scorio";

console.log(metrics.bayes([0, 1, 1]));
console.log(rank.avg([[1, 1], [0, 1]]).ranking);
console.log(aggregate.majorityVote(["A", "A", "B"]));
```

`eval` is a reserved word in some JavaScript contexts, so give it a local name
such as `metrics` when importing it. CommonJS callers can use the same subpaths:

```cjs
const { bayes } = require("scorio/eval");

console.log(bayes([0, 1, 1]));
```

The supported subpaths are `scorio/eval`, `scorio/rank`, `scorio/aggregate`,
`scorio/sinf`, and `scorio/utils`. Import public functions from these paths or
from the root namespaces.

## Use the repository version

The npm release may be older than these docs. To use the current source, clone
the repository and build the package:

```sh
git clone https://github.com/mohsenhariri/scorio.git
cd scorio/js/scorio
npm ci
npm run build
```

Then install the built directory from your application's directory. Adjust the
relative path to your checkout:

```sh
npm install ../scorio/js/scorio
```

For a tarball you can copy to another machine, run `npm pack` in `js/scorio`
after building. Install the resulting `.tgz` file with `npm install`.

## Names and types

CamelCase names are used in the guides. Snake_case aliases refer to the same
functions:

```ts
import { passAtK, pass_at_k, type Matrix } from "scorio/eval";

const R: Matrix = [[0, 1, 1], [1, 0, 1]];
console.log(passAtK(R, 2) === pass_at_k(R, 2)); // => true
```

Types are exported from their API subpath. TypeScript resolves them through the
package's `exports` map; no separate `@types/scorio` package is needed. Use a
resolution mode that understands package exports, such as `NodeNext` in a
Node.js project or `Bundler` in a bundled app.

The reference also shows a few types under “Referenced types” to explain
parameter and return shapes. Those types are used by public functions but are
not named exports you can import from the package.
