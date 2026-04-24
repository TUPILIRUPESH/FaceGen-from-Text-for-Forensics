import java.util.Scanner;
import java.util.InputMismatchException;

class Exception {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        try {
            System.out.println("enter the first decimal number");
            int firstnum = sc.nextInt();

            System.out.println("enter the second decimal number");
            int secondnum = sc.nextInt();

            int result = firstnum / secondnum;

            System.out.println("Result =  " + result);

        } catch (ArithmeticException e) {
            System.out.println("Do not enter the 0 into the input ");
        } catch (InputMismatchException e) {
            System.out.println("please enter only integers");
        } catch (NullPointerException e) {
            System.out.println("please enter the values");
        }
    }
}