
from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.data.synthetic_invoices import SUPPORTED_CURRENCIES


class InvoiceLineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    line_total: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def validate_line_total(self):
        expected = (self.quantity * self.unit_price).quantize(
            Decimal("0.01")
        )
        actual = self.line_total.quantize(Decimal("0.01"))

        if expected != actual:
            raise ValueError(
                "line_total must equal quantity multiplied by unit_price"
            )
        return self


class InvoicePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: str | None
    invoice_date: date | None
    due_date: date | None
    supplier_name: str | None
    customer_name: str | None
    currency: str | None
    subtotal: Decimal | None
    tax_amount: Decimal | None
    total_amount: Decimal | None
    line_items: list[InvoiceLineItem]

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value):
        if value is not None and value not in SUPPORTED_CURRENCIES:
            raise ValueError("Currency is outside the configured pilot allowlist")
        return value

    @model_validator(mode="after")
    def validate_financial_consistency(self):
        for field_name in ("subtotal", "tax_amount", "total_amount"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")

        if self.invoice_date and self.due_date:
            if self.due_date < self.invoice_date:
                raise ValueError("due_date cannot precede invoice_date")

        if self.subtotal is not None:
            line_sum = sum(
                (item.line_total for item in self.line_items),
                Decimal("0.00"),
            )
            if line_sum.quantize(Decimal("0.01")) != self.subtotal.quantize(
                Decimal("0.01")
            ):
                raise ValueError("subtotal must equal the sum of line items")

        if (
            self.subtotal is not None
            and self.tax_amount is not None
            and self.total_amount is not None
        ):
            expected_total = self.subtotal + self.tax_amount
            if expected_total.quantize(Decimal("0.01")) != self.total_amount.quantize(
                Decimal("0.01")
            ):
                raise ValueError("total_amount must equal subtotal plus tax")

        return self
