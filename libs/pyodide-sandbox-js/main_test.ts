import { assertEquals } from "@std/assert";
import { runPython } from "./main.ts";

Deno.test("runPython simple test", async () => {
  const result = await runPython("x = 2 + 3; x", {});
  assertEquals(result.success, true);
  assertEquals(JSON.parse(result.jsonResult || "null"), 5);
});

Deno.test("runPython with error", async () => {
  const result = await runPython("x = 1/0", {});
  assertEquals(result.success, false);
});
