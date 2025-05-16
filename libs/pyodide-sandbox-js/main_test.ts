import { assertEquals, assertNotEquals, assertExists } from "@std/assert";
import { runPython } from "./main.ts";

Deno.test("runPython simple test", async () => {
  const result = await runPython("x = 2 + 3; x", {});
  assertEquals(result.success, true);
  assertEquals(JSON.parse(result.jsonResult || "null"), 5);
});

Deno.test("runPython with stdout", async () => {
  const result = await runPython("x = 5; print(x); x", {});
  assertEquals(result.success, true);
  assertEquals(result.stdout?.join(''), "5");
  assertEquals(JSON.parse(result.jsonResult || "null"), 5);
  assertEquals(result.stderr?.length, 0);
});

Deno.test("runPython with error - division by zero", async () => {
  const result = await runPython("x = 1/0", {});
  assertEquals(result.success, false);
  assertNotEquals(result.error?.length, 0);
  assertEquals(result.error?.includes("ZeroDivisionError"), true);
});

Deno.test("runPython with error - syntax error", async () => {
  const result = await runPython("x = 5; y = x +", {});
  assertEquals(result.success, false);
  assertNotEquals(result.error?.length, 0);
  // Check that error contains SyntaxError
  assertEquals(result.error?.includes("SyntaxError"), true);
});

Deno.test("runPython with error - name error", async () => {
  const result = await runPython("undefined_variable", {});
  assertEquals(result.success, false);
  assertExists(result.error);
  // Check that error contains NameError
  assertEquals(result.error?.includes("NameError"), true);
});
