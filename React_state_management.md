# React State Management

## 1. Principle of Least Knowledge(Law of Demeter)

### 1.1 Remove any non-essential state variables
> **“A state variable is only necessary to keep information between re-renders of a component. Within a single event handler, a regular variable will do fine. Don’t introduce state variables when a regular variable works well.”** 
> — the official React documentation: https://react.dev/learn/state-a-components-memory#recap

```js
// ❌ Bad Example: Two Forms of Unnecessary State in One Component
function UserDashboard({ user, price, taxRate }) {
  // Derived value stored as state → can cause infinite re-renders
  const [total, setTotal] = useState(price * taxRate);

  useEffect(() => {
    setTotal(price * taxRate); // triggers render on every render → loop risk
  });

  // Copying props into state → becomes stale when props update
  const [name, setName] = useState(user.name); // will NOT update if user changes

  return (
    <div>
      <h3>User: {name}</h3>          {/* may show outdated name */}
      <h3>Total: {total}</h3>        {/* computed via repeated state updates */}
    </div>
  );
}

```
```js
// ✔ Good Example: No Redundant State — Always Correct & Predictable
function UserDashboard({ user, price, taxRate }) {
  // Compute derived values directly
  const total = price * taxRate;

  // Read props directly to avoid stale state
  const name = user.name;

  return (
    <div>
      <h3>User: {name}</h3>
      <h3>Total: {total}</h3>
    </div>
  );
}

```

### 1.2 Fewer Parameters Lead to Better Components and Functions
> **“Keep it simple. Complexity is the enemy.”**  
> — *Linus Torvalds*

```js
// ❌ Bad: too many props, some redundant, some not owned by this component
function OrderSummary({
  showDebugInfo,          // Not necessary
  totalPrice,             // Can be derived from items
  currency, locale,       // Should come from global settings
  itemNames,              // Should be grouped as a single data structure
  itemQuantities,
  itemPrices
}) {
  return (
    <div>
      <h2>Order Summary</h2>
      <!-- rendering logic -->
    </div>
  );
}

// Usage: caller must manually compute & pass everything
OrderSummary({
  showDebugInfo: process.env.NODE_ENV === "development",
  totalPrice: cart.items.reduce((s, i) => s + i.price * i.quantity, 0),
  currency: window.appSettings.currency,
  locale: window.appSettings.locale,
  itemNames: cart.items.map(i => i.name),
  itemQuantities: cart.items.map(i => i.quantity),
  itemPrices: cart.items.map(i => i.price)
});
```
```js
// Global settings can be accessed directly
const settings = {
  currency: "USD",
  locale: "en-US"
};

// ✔ Good: only necessary, minimal props
function OrderSummary({ items }) {
  const currency = settings.currency;
  const locale = settings.locale;

  // Derive instead of passing manually
  const totalPrice = items.reduce(
    (sum, item) => sum + item.price * item.quantity,
    0
  );

  return (
    <div>
      <h2>Order Summary</h2>
      <ul>
        ${items
          .map(
            item =>
              `<li>${item.name} × ${item.quantity} — ${item.price.toFixed(
                2
              )} ${currency}</li>`
          )
          .join("")}
      </ul>
      <p>Total: ${totalPrice.toFixed(2)} ${currency} (${locale})</p>
    </div>
  );
}

// Usage: cleaner, more meaningful, fewer responsibilities
OrderSummary({
  items: cart.items   // Pass grouped domain data instead of parallel arrays
});
```

## 2. Use Immer to manage form status

### 2.1 Treat state as read-only
> — the official React documentation: https://react.dev/learn/updating-objects-in-state#treat-state-as-read-only

```js
const initialData = {
  items: ["A", "B", "C"],
  pagination: {
    page: 1,
    pageSize: 10,
    total: 30,
  },
};

export default function ItemList() {
  const [listState, setListState] = useState(initialData);

  const goToNextPage = () => {
    // ❌ Risky: mutating the pagination object directly
    // This changes the nested object in-place.
    listState.pagination.page += 1;

    // ✅ Top-level object is copied, so React will still re-render
    setListState({
      ...listState,
    });
  };

  const { items, pagination } = listState;

  return (
    <div>
      <h2>Items</h2>
      <ul>
        {items.map((it) => (
          <li key={it}>{it}</li>
        ))}
      </ul>
      <p>
        Page: {pagination.page} / {Math.ceil(pagination.total / pagination.pageSize)}
      </p>
      <button onClick={goToNextPage}>Next page</button>
    </div>
  );
}
```
```js
const initialData = {
  items: ["A", "B", "C"],
  pagination: {
    page: 1,
    pageSize: 10,
    total: 30,
  },
};

export default function ItemList() {
  const [listState, setListState] = useState(initialData);

  const { items, pagination } = listState;

  useEffect(() => {
    // ❌ Bug: this effect depends on the *reference* of pagination.
    // If we mutate pagination in-place, the reference never changes,
    // so this effect will NOT run on page changes.
    fetchItems(pagination).then((newItems) => {
      setListState((prev) => ({
        ...prev,
        items: newItems,
      }));
    });
  }, [pagination]); // <-- depends on pagination object identity

  const goToNextPage = () => {
    // ❌ Risky: mutating the pagination object directly
    // This changes the nested object in-place.
    listState.pagination.page += 1;

    // ✅ Top-level object is copied, so React will still re-render
    setListState({
      ...listState,
    });
  };

  return (
    <div>
      <h2>Items</h2>
      <ul>
        {items.map((it) => (
          <li key={it}>{it}</li>
        ))}
      </ul>
      <p>
        Page: {pagination.page} / {Math.ceil(pagination.total / pagination.pageSize)}
      </p>
      <button onClick={goToNextPage}>Next page</button>
    </div>
  );
}
```

### 2.2 Write concise update logic with Immer
> — the official React documentation: https://react.dev/learn/updating-objects-in-state#write-concise-update-logic-with-immer

How does Immer work?

The draft provided by Immer is a special type of object, called a Proxy, that “records” what you do with it. This is why you can mutate it freely as much as you like! Under the hood, Immer figures out which parts of the draft have been changed, and produces a completely new object that contains your edits.
```js
updatePerson(draft => {
  draft.artwork.city = 'Lagos';
});
```

### 2.3 useFormData：Immutable Form State with Auto-Restore

- Built on **useImmer** for intuitive form updates  
- Automatically **restores unchanged objects**  
- Simple & fast change detection, no deep compare needed
- Usage:
```ts
const [formData, updateFormData] = useFormData(initData);
const isChanged = formData !== initData
```

## 3. Manage server-side status

### 3.1 TanStack Query
> — The official React docs recommend TanStack Query as a solution for managing server state and data fetching: https://react.dev/learn/build-a-react-app-from-scratch#data-fetching

```js
export default function UserSearch({ keyword }) {
  const [result, setResult] = useState(null);

  useEffect(() => {
    // First request (slow)
    fetch(`/api/search?q=${keyword}`)
      .then(res => res.json())
      .then(data => {
        // ❌ Danger: this may run AFTER a later request
        setResult(data);
      });

    // Then maybe you trigger another effect because keyword changes,
    // and the second fetch returns faster.
  }, [keyword]);

  return (
    <div>
      <p>Keyword: {keyword}</p>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </div>
  );
}
```

