import React, {
  forwardRef,
  useCallback,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

const DEFAULT_FORMAT_CHARS = {
  '9': '[0-9]',
  a: '[A-Za-z]',
  '*': '[A-Za-z0-9]',
};

function assignRef(ref, value) {
  if (!ref) return;
  if (typeof ref === 'function') {
    ref(value);
  } else {
    ref.current = value;
  }
}

function parseMask(mask, formatChars) {
  const tokens = [];
  let escaped = false;
  let slotIndex = 0;

  for (let i = 0; i < mask.length; i += 1) {
    const ch = mask[i];

    if (!escaped && ch === '\\') {
      escaped = true;
      continue;
    }

    if (!escaped && Object.prototype.hasOwnProperty.call(formatChars, ch)) {
      tokens.push({
        type: 'slot',
        maskChar: ch,
        slotIndex,
        regexp: new RegExp(`^${formatChars[ch]}$`),
      });
      slotIndex += 1;
    } else {
      tokens.push({
        type: 'fixed',
        char: ch,
      });
    }

    escaped = false;
  }

  if (escaped) {
    tokens.push({
      type: 'fixed',
      char: '\\',
    });
  }

  const slotCount = slotIndex;
  const slotIndexToMaskedIndex = new Array(slotCount);
  const slotRegexes = new Array(slotCount);

  let slotsBefore = 0;
  const slotCountBeforeToken = new Array(tokens.length);

  for (let i = 0; i < tokens.length; i += 1) {
    slotCountBeforeToken[i] = slotsBefore;
    if (tokens[i].type === 'slot') {
      slotIndexToMaskedIndex[slotsBefore] = i;
      slotRegexes[slotsBefore] = tokens[i].regexp;
      slotsBefore += 1;
    }
  }

  let firstSlotMaskedIndex = tokens.length;
  for (let i = 0; i < tokens.length; i += 1) {
    if (tokens[i].type === 'slot') {
      firstSlotMaskedIndex = i;
      break;
    }
  }

  return {
    tokens,
    slotCount,
    slotRegexes,
    slotIndexToMaskedIndex,
    firstSlotMaskedIndex,
    maskLength: tokens.length,
    slotCountBeforeToken,
  };
}

function extractRaw(value, parsed) {
  if (!value) return '';

  const { tokens } = parsed;
  const raw = [];

  let tokenIndex = 0;
  let valueIndex = 0;

  while (tokenIndex < tokens.length && valueIndex < value.length) {
    const token = tokens[tokenIndex];
    const ch = value[valueIndex];

    if (token.type === 'fixed') {
      if (ch === token.char) {
        valueIndex += 1;
      }
      tokenIndex += 1;
      continue;
    }

    if (token.regexp.test(ch)) {
      raw.push(ch);
      tokenIndex += 1;
      valueIndex += 1;
    } else {
      valueIndex += 1;
    }
  }

  return raw.join('');
}

function normalizeRaw(raw, parsed) {
  const chars = Array.from(raw || '');
  const result = [];
  let sourceIndex = 0;

  for (let slot = 0; slot < parsed.slotCount; slot += 1) {
    const regexp = parsed.slotRegexes[slot];

    while (sourceIndex < chars.length && !regexp.test(chars[sourceIndex])) {
      sourceIndex += 1;
    }

    if (sourceIndex >= chars.length) break;

    result.push(chars[sourceIndex]);
    sourceIndex += 1;
  }

  return result.join('');
}

function formatMaskedValue(raw, parsed, maskChar) {
  const { tokens, slotCountBeforeToken } = parsed;
  const rawChars = Array.from(raw || '');
  const useMaskChar = maskChar !== null && maskChar !== undefined && maskChar !== '';
  const result = [];

  for (let i = 0; i < tokens.length; i += 1) {
    const token = tokens[i];

    if (token.type === 'fixed') {
      if (useMaskChar) {
        result.push(token.char);
      } else {
        const filledSlotsBefore = slotCountBeforeToken[i];
        if (filledSlotsBefore === 0 || rawChars.length >= filledSlotsBefore) {
          result.push(token.char);
        }
      }
      continue;
    }

    const slot = token.slotIndex;
    if (slot < rawChars.length) {
      result.push(rawChars[slot]);
    } else if (useMaskChar) {
      result.push(maskChar);
    }
  }

  return result.join('');
}

function getNextSlotIndexFromMaskedPos(maskedPos, parsed) {
  const { slotIndexToMaskedIndex, slotCount } = parsed;
  for (let slot = 0; slot < slotCount; slot += 1) {
    if (slotIndexToMaskedIndex[slot] >= maskedPos) return slot;
  }
  return slotCount;
}

function getPrevSlotIndexBeforeMaskedPos(maskedPos, parsed) {
  const { slotIndexToMaskedIndex, slotCount } = parsed;
  for (let slot = slotCount - 1; slot >= 0; slot -= 1) {
    if (slotIndexToMaskedIndex[slot] < maskedPos) return slot;
  }
  return -1;
}

function getCaretPosForSlot(slotIndex, parsed, maskedValue) {
  if (slotIndex >= parsed.slotCount) {
    return maskedValue.length;
  }
  const maskedPos = parsed.slotIndexToMaskedIndex[slotIndex];
  return Math.max(0, Math.min(maskedPos, maskedValue.length));
}

function countFilledSlotsBeforeMaskedPos(maskedPos, parsed, domValue) {
  const slice = (domValue || '').slice(0, maskedPos);
  return extractRaw(slice, parsed).length;
}

function refitFrom(startSlot, prefixRaw, sourceChars, parsed) {
  const result = Array.from(prefixRaw);
  let sourceIndex = 0;

  for (let slot = startSlot; slot < parsed.slotCount; slot += 1) {
    const regexp = parsed.slotRegexes[slot];

    while (sourceIndex < sourceChars.length && !regexp.test(sourceChars[sourceIndex])) {
      sourceIndex += 1;
    }

    if (sourceIndex >= sourceChars.length) break;

    result.push(sourceChars[sourceIndex]);
    sourceIndex += 1;
  }

  return result.join('');
}

function applySelectionReplace(raw, insertText, selectionStart, selectionEnd, parsed) {
  const rawChars = Array.from(raw || '');
  const startSlot = getNextSlotIndexFromMaskedPos(selectionStart, parsed);
  const endSlot = getNextSlotIndexFromMaskedPos(selectionEnd, parsed);

  const prefix = rawChars.slice(0, startSlot);
  const tail = rawChars.slice(endSlot);
  const sourceChars = [...Array.from(insertText || ''), ...tail];

  const nextRaw = refitFrom(startSlot, prefix, sourceChars, parsed);

  return {
    raw: nextRaw,
    caretSlot: Math.min(startSlot + normalizeRaw(insertText || '', {
      ...parsed,
      slotCount: parsed.slotCount - startSlot,
      slotRegexes: parsed.slotRegexes.slice(startSlot),
    }).length, parsed.slotCount),
    startSlot,
  };
}

function createChangeLikeEvent(input, nativeEvent) {
  return {
    type: 'change',
    target: input,
    currentTarget: input,
    nativeEvent: nativeEvent || null,
    preventDefault() {
      if (nativeEvent && typeof nativeEvent.preventDefault === 'function') {
        nativeEvent.preventDefault();
      }
    },
    stopPropagation() {
      if (nativeEvent && typeof nativeEvent.stopPropagation === 'function') {
        nativeEvent.stopPropagation();
      }
    },
  };
}

const MaskedInput = forwardRef(function MaskedInput(props, forwardedRef) {
  const {
    mask,
    maskChar = '_',
    formatChars,
    value,
    defaultValue,
    onChange,
    onBeforeInput,
    onKeyDown,
    onPaste,
    onFocus,
    onClick,
    ...rest
  } = props;

  const mergedFormatChars = useMemo(
    () => ({
      ...DEFAULT_FORMAT_CHARS,
      ...(formatChars || {}),
    }),
    [formatChars]
  );

  const parsed = useMemo(
    () => parseMask(String(mask || ''), mergedFormatChars),
    [mask, mergedFormatChars]
  );

  const isControlled = value !== undefined && value !== null;

  const [innerRaw, setInnerRaw] = useState(() =>
    normalizeRaw(extractRaw(String(defaultValue ?? value ?? ''), parsed), parsed)
  );

  const inputRef = useRef(null);
  const pendingSelectionRef = useRef(null);

  const setRefs = useCallback(
    (node) => {
      inputRef.current = node;
      assignRef(forwardedRef, node);
    },
    [forwardedRef]
  );

  const currentRaw = isControlled
    ? normalizeRaw(extractRaw(String(value ?? ''), parsed), parsed)
    : normalizeRaw(innerRaw, parsed);

  const maskedValue = useMemo(
    () => formatMaskedValue(currentRaw, parsed, maskChar),
    [currentRaw, parsed, maskChar]
  );

  useLayoutEffect(() => {
    const input = inputRef.current;
    if (!input) return;

    if (pendingSelectionRef.current) {
      const { start, end } = pendingSelectionRef.current;
      const safeStart = Math.max(0, Math.min(start, input.value.length));
      const safeEnd = Math.max(0, Math.min(end, input.value.length));
      try {
        input.setSelectionRange(safeStart, safeEnd);
      } catch (_) {}
      pendingSelectionRef.current = null;
    }
  }, [maskedValue]);

  const commitRaw = useCallback(
    (nextRaw, caretPos, nativeEvent) => {
      const normalized = normalizeRaw(nextRaw, parsed);
      const nextMasked = formatMaskedValue(normalized, parsed, maskChar);

      if (!isControlled) {
        setInnerRaw(normalized);
      }

      const input = inputRef.current;
      if (input) {
        input.value = nextMasked;
      }

      pendingSelectionRef.current = {
        start: caretPos,
        end: caretPos,
      };

      if (onChange && input) {
        onChange(createChangeLikeEvent(input, nativeEvent));
      }
    },
    [isControlled, maskChar, onChange, parsed]
  );

  const handleBeforeInput = useCallback(
    (e) => {
      if (onBeforeInput) onBeforeInput(e);
      if (e.defaultPrevented) return;
      if (!parsed.slotCount) return;

      const native = e.nativeEvent;
      const isComposing = native && native.isComposing;
      if (isComposing) return;

      const data = native && typeof native.data === 'string' ? native.data : null;
      const inputType = native && native.inputType ? native.inputType : '';

      if (!data) return;
      if (!inputType.startsWith('insert')) return;

      const input = inputRef.current;
      if (!input) return;

      const selectionStart = input.selectionStart ?? 0;
      const selectionEnd = input.selectionEnd ?? selectionStart;

      e.preventDefault();

      const result = applySelectionReplace(
        currentRaw,
        data,
        selectionStart,
        selectionEnd,
        parsed
      );

      const nextMasked = formatMaskedValue(result.raw, parsed, maskChar);
      const caretPos = getCaretPosForSlot(result.caretSlot, parsed, nextMasked);

      commitRaw(result.raw, caretPos, native);
    },
    [commitRaw, currentRaw, maskChar, onBeforeInput, parsed]
  );

  const handlePaste = useCallback(
    (e) => {
      if (onPaste) onPaste(e);
      if (e.defaultPrevented) return;
      if (!parsed.slotCount) return;

      const text = e.clipboardData?.getData('text') ?? '';
      const input = inputRef.current;
      if (!input) return;

      e.preventDefault();

      const selectionStart = input.selectionStart ?? 0;
      const selectionEnd = input.selectionEnd ?? selectionStart;

      const result = applySelectionReplace(
        currentRaw,
        text,
        selectionStart,
        selectionEnd,
        parsed
      );

      const nextMasked = formatMaskedValue(result.raw, parsed, maskChar);
      const caretPos = getCaretPosForSlot(result.caretSlot, parsed, nextMasked);

      commitRaw(result.raw, caretPos, e.nativeEvent);
    },
    [commitRaw, currentRaw, onPaste, parsed, maskChar]
  );

  const handleKeyDown = useCallback(
    (e) => {
      if (onKeyDown) onKeyDown(e);
      if (e.defaultPrevented) return;
      if (!parsed.slotCount) return;

      const input = inputRef.current;
      if (!input) return;

      const selectionStart = input.selectionStart ?? 0;
      const selectionEnd = input.selectionEnd ?? selectionStart;
      const hasSelection = selectionStart !== selectionEnd;
      const rawChars = Array.from(currentRaw);

      if (e.key === 'Backspace') {
        e.preventDefault();

        if (hasSelection) {
          const startSlot = getNextSlotIndexFromMaskedPos(selectionStart, parsed);
          const endSlot = getNextSlotIndexFromMaskedPos(selectionEnd, parsed);
          const nextRaw = refitFrom(
            startSlot,
            rawChars.slice(0, startSlot),
            rawChars.slice(endSlot),
            parsed
          );
          const nextMasked = formatMaskedValue(nextRaw, parsed, maskChar);
          const caretPos = getCaretPosForSlot(startSlot, parsed, nextMasked);
          commitRaw(nextRaw, caretPos, e.nativeEvent);
          return;
        }

        let deleteSlot = getPrevSlotIndexBeforeMaskedPos(selectionStart, parsed);
        deleteSlot = Math.min(deleteSlot, rawChars.length - 1);

        if (deleteSlot < 0) return;

        const nextRaw = refitFrom(
          deleteSlot,
          rawChars.slice(0, deleteSlot),
          rawChars.slice(deleteSlot + 1),
          parsed
        );
        const nextMasked = formatMaskedValue(nextRaw, parsed, maskChar);
        const caretPos = getCaretPosForSlot(deleteSlot, parsed, nextMasked);
        commitRaw(nextRaw, caretPos, e.nativeEvent);
        return;
      }

      if (e.key === 'Delete') {
        e.preventDefault();

        if (hasSelection) {
          const startSlot = getNextSlotIndexFromMaskedPos(selectionStart, parsed);
          const endSlot = getNextSlotIndexFromMaskedPos(selectionEnd, parsed);
          const nextRaw = refitFrom(
            startSlot,
            rawChars.slice(0, startSlot),
            rawChars.slice(endSlot),
            parsed
          );
          const nextMasked = formatMaskedValue(nextRaw, parsed, maskChar);
          const caretPos = getCaretPosForSlot(startSlot, parsed, nextMasked);
          commitRaw(nextRaw, caretPos, e.nativeEvent);
          return;
        }

        const deleteSlot = getNextSlotIndexFromMaskedPos(selectionStart, parsed);
        if (deleteSlot >= rawChars.length) return;

        const nextRaw = refitFrom(
          deleteSlot,
          rawChars.slice(0, deleteSlot),
          rawChars.slice(deleteSlot + 1),
          parsed
        );
        const nextMasked = formatMaskedValue(nextRaw, parsed, maskChar);
        const caretPos = getCaretPosForSlot(deleteSlot, parsed, nextMasked);
        commitRaw(nextRaw, caretPos, e.nativeEvent);
      }
    },
    [commitRaw, currentRaw, maskChar, onKeyDown, parsed]
  );

  const handleChange = useCallback(
    (e) => {
      if (!parsed.slotCount) {
        if (onChange) onChange(e);
        return;
      }

      const domValue = e.target.value;
      const nextRaw = normalizeRaw(extractRaw(domValue, parsed), parsed);
      const nextMasked = formatMaskedValue(nextRaw, parsed, maskChar);

      if (!isControlled) {
        setInnerRaw(nextRaw);
      }

      const input = inputRef.current;
      const domSelectionStart = input?.selectionStart ?? domValue.length;
      const filledSlotsBeforeCaret = countFilledSlotsBeforeMaskedPos(
        domSelectionStart,
        parsed,
        domValue
      );
      const caretPos = getCaretPosForSlot(filledSlotsBeforeCaret, parsed, nextMasked);

      if (input && input.value !== nextMasked) {
        input.value = nextMasked;
      }

      pendingSelectionRef.current = {
        start: caretPos,
        end: caretPos,
      };

      if (onChange) onChange(e);
    },
    [isControlled, maskChar, onChange, parsed]
  );

  const handleFocus = useCallback(
    (e) => {
      if (onFocus) onFocus(e);

      if (!parsed.slotCount) return;
      const input = inputRef.current;
      if (!input) return;

      const pos = input.selectionStart ?? 0;
      const nextSlot = getNextSlotIndexFromMaskedPos(pos, parsed);
      const caretPos = getCaretPosForSlot(nextSlot, parsed, maskedValue);

      pendingSelectionRef.current = {
        start: caretPos,
        end: caretPos,
      };
    },
    [maskedValue, onFocus, parsed]
  );

  const handleClick = useCallback(
    (e) => {
      if (onClick) onClick(e);

      if (!parsed.slotCount) return;
      const input = inputRef.current;
      if (!input) return;

      const pos = input.selectionStart ?? 0;
      const nextSlot = getNextSlotIndexFromMaskedPos(pos, parsed);
      const caretPos = getCaretPosForSlot(nextSlot, parsed, maskedValue);

      pendingSelectionRef.current = {
        start: caretPos,
        end: caretPos,
      };
    },
    [maskedValue, onClick, parsed]
  );

  if (!mask) {
    return (
      <input
        {...rest}
        ref={setRefs}
        value={value}
        defaultValue={defaultValue}
        onChange={onChange}
        onBeforeInput={onBeforeInput}
        onKeyDown={onKeyDown}
        onPaste={onPaste}
        onFocus={onFocus}
        onClick={onClick}
      />
    );
  }

  return (
    <input
      {...rest}
      ref={setRefs}
      value={maskedValue}
      onChange={handleChange}
      onBeforeInput={handleBeforeInput}
      onKeyDown={handleKeyDown}
      onPaste={handlePaste}
      onFocus={handleFocus}
      onClick={handleClick}
    />
  );
});

export default MaskedInput;
